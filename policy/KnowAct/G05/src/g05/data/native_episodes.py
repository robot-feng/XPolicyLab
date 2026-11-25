"""Build same-Episode Native Context samples from the existing LeRobot dataset.

The sampler uses absolute frame indices and never crosses an Episode boundary.
It does not sample demonstrations from other Episodes; cross-Episode ICL is a
separate, not-yet-implemented data path.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch


def history_indices(
    start: int,
    end: int,
    current: int,
    stride: int,
    turns: int,
) -> list[int]:
    """Return chronological, same-Episode history indices before ``current``."""
    if not start <= current < end or stride < 1 or turns < 1:
        raise ValueError("invalid episode bounds")

    available_turns = min(turns - 1, (current - start) // stride)
    first = current - available_turns * stride
    return list(range(first, current, stride))


def hold_pad(action: torch.Tensor, count: int) -> torch.Tensor:
    """Keep the applied prefix and repeat its last value to the fixed horizon.

    This is a representation adapter for a fixed-horizon ActionCodec. The padded
    suffix is not claimed to be measured or executed and must never be sent to
    the robot as part of the execution receipt.
    """
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= len(action)
    ):
        raise ValueError("invalid executed action prefix")

    result = action.clone()
    result[count:] = result[count - 1 : count]
    return result


def validate_history_processor(processor: Any) -> None:
    """Enforce the currently verified pointwise action-normalization path."""
    if processor is None:
        return

    if (
        getattr(processor, "action_state_transforms", None)
        or getattr(processor, "use_stepwise_action_norm", False)
        or getattr(processor, "_vlm_input_action_normalizer", None) is not None
    ):
        raise ValueError(
            "executed history requires pointwise identical normalization and no action transforms"
        )

    if hasattr(processor, "action_filter"):
        from g05.data_processor.transforms.action_filter import DummyActionFilter

        if not isinstance(processor.action_filter, DummyActionFilter):
            raise ValueError("executed history requires the dummy action filter")


def completed_history(
    dataset: Any,
    current: int,
    turns: int,
    replan_steps: int = 0,
) -> list[dict[str, Any]]:
    """Load completed same-Episode history for one already-selected row.

    The ``current`` index is the absolute dataset index from the selected sample.
    The history loader bypasses weighted/random index remapping and loads those
    exact absolute indices. Positive observation offsets and non-zero action
    offsets are rejected to avoid future leakage or misaligned action history.
    """
    if turns <= 1:
        return []
    if dataset.obs_size != 1 or dataset.past_action_size != 0:
        raise ValueError("native episodes require single current observations")

    image_meta = dataset.image_meta or []
    if any(
        meta.get("time_offset", 0) > 0
        for meta in [*dataset.state_meta, *image_meta]
    ):
        raise ValueError("future observations forbidden")
    if any(meta.get("time_offset", 0) != 0 for meta in dataset.action_meta):
        raise ValueError("native episodes currently require zero action time offsets")
    if not math.isclose(float(dataset.model_fps), float(dataset.fps)):
        raise ValueError("native episode stride requires equal data/control fps")

    episode_starts = dataset.episode_data_index["from"]
    episode_ends = dataset.episode_data_index["to"]
    episode_index = int(
        torch.searchsorted(episode_ends, torch.tensor(current), right=True)
    )
    start = max(int(episode_starts[episode_index]), int(dataset._start_idx))
    end = min(int(episode_ends[episode_index]), int(dataset._end_idx))

    horizon = int(dataset.action_size)
    stride = int(replan_steps) if replan_steps else horizon
    if not 1 <= stride <= horizon:
        raise ValueError("replan steps must fit action horizon")
    if replan_steps:
        validate_history_processor(getattr(dataset, "processor", None))

    indices = history_indices(start, end, current, stride, turns)
    history: list[dict[str, Any]] = []
    for absolute_index in indices:
        sample = dataset.load_native_history(absolute_index)
        if int(sample["idx"]) != absolute_index or "action_is_pad" not in sample:
            raise RuntimeError("history loader returned a different or incomplete sample")

        action_is_pad = torch.as_tensor(sample["action_is_pad"])
        if bool(action_is_pad[:stride].any()):
            # Do not present a padded/unexecuted prefix as completed history.
            history = []
            continue

        if replan_steps:
            action_entry = sample["samples"]["action"]
            padded_action = hold_pad(action_entry["value"], stride)
            padded_raw_action = hold_pad(sample["action"], stride)
            sample = {
                **sample,
                "samples": {
                    **sample["samples"],
                    "action": {**action_entry, "value": padded_action},
                },
                "action": padded_raw_action,
                "_native_executed_steps": stride,
            }

            # Legacy gt_action is not used for CE. Do not retain its unseen suffix.
            if isinstance(sample.get("gt_action"), torch.Tensor):
                sample["gt_action"] = hold_pad(sample["gt_action"], stride)

            padded_mask = action_is_pad.clone()
            padded_mask[stride:] = True
            sample["action_is_pad"] = padded_mask

        history.append(sample)

    return history


def collate_episodes(
    batch: list[dict[str, Any]],
    collate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Attach chronological B=1 turns while retaining the legacy current batch.

    The supplied G05 collate function is reused unchanged. In executed-history
    mode, history rows carry ``_native_executed_steps`` and are marked
    context-only. With ``replan_steps=0``, historical rows are not automatically
    masked and may remain supervised. Each following turn receives the number
    of control steps applied from the previous action proposal when available.
    """
    current_rows: list[dict[str, Any]] = []
    episodes: list[list[dict[str, Any]]] = []

    for item in batch:
        current = dict(item)
        history = current.pop("native_history", [])
        turns = [*history, current]
        episode: list[dict[str, Any]] = []

        for index, turn in enumerate(turns):
            row = dict(turn)
            executed_steps = row.pop("_native_executed_steps", None)
            mini_batch = collate([row])

            if executed_steps is not None:
                mini_batch["supervise"] = False

            if index:
                previous_turn = turns[index - 1]
                previous_steps = previous_turn.get(
                    "_native_executed_steps",
                    previous_turn["action"].shape[0],
                )
                mini_batch["executed_steps"] = int(previous_steps)

            episode.append(mini_batch)

        current_rows.append(current)
        episodes.append(episode)

    result = collate(current_rows)
    result["native_episodes"] = episodes
    return result
