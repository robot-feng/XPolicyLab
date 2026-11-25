"""Native multimodal context for KnowAct.

This module reuses G05's Processor, Qwen3.5 backbone, ActionCodec grammar and
LM head. It adds no trainable module. Each stream owns one committed
``InferenceState``; the first supported action protocol is fixed-layout BAR16.
Partial RVQ decoding is intentionally not supported.
"""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import groupby
from typing import Any, Iterable

import torch

from .helpers.proprio_helper import build_proprio_batch
from .io.input_preprocessor import TOKEN_INDEX


def active(policy: Any) -> bool:
    """Whether Native Context is enabled in this policy configuration."""
    return bool(policy.config.get("native_context", False))


def _budget(policy: Any, size: int) -> None:
    """Reject over-budget context rather than silently evicting history."""
    maximum = int(policy.config.get("native_context_max_tokens", 16384))
    if size > maximum:
        raise RuntimeError(
            "Native context budget exceeded; reset/replay explicitly "
            "(no silent eviction)"
        )


def execution_history(policy: Any) -> bool:
    """Whether training/inference records only confirmed executed action prefixes."""
    return int(policy.config.get("native_replan_steps", 0)) > 0


def receipt(
    sample: dict[str, Any],
    count: int | None,
    *,
    executed_only: bool = False,
) -> dict[str, Any]:
    """Add an execution receipt through the existing command-text slot.

    The receipt reports the number of control commands applied by the
    environment. It is not a measured robot trajectory; measured state arrives
    in the current observation.
    """
    if count is None:
        return sample
    if int(count) != count or count < 0:
        raise ValueError("execution receipt must be a nonnegative integer")
    if "command" not in sample:
        raise ValueError("execution receipt requires the native command slot")

    template = sample.get("template")
    command_pattern = r"<command_text(?:_!_(?:\d+)|_!)?>"
    if not template or not re.search(command_pattern, template):
        raise ValueError("execution receipt requires an explicit command template")

    template = re.sub(
        f"({command_pattern})",
        r"\1\n<executionreceipt_text_!>",
        template,
        count=1,
    )
    if executed_only:
        receipt_text = (
            f"Previous action history: {int(count)} control steps applied; "
            "remaining positions are padding. Use the current measured state."
        )
    else:
        receipt_text = (
            f"Previous plan: {int(count)} control steps applied. "
            "Use the current measured state."
        )
    return {
        **sample,
        "template": template,
        "executionreceipt": receipt_text,
    }


def encode(
    policy: Any,
    sample: dict[str, Any],
    pixels: Any,
    *,
    training: bool = False,
    prefix: bool = False,
):
    """Encode one unpadded multimodal turn with the original G05 encoders."""
    device = policy.device
    if isinstance(pixels, dict):
        pixels = {key: value.to(device) for key, value in pixels.items()}
    else:
        pixels = pixels.to(device)
    pixels = policy.process_pixel_values(pixels)

    processor_sample = (
        {key: value for key, value in sample.items() if key != "action"}
        if prefix
        else sample
    )
    ids, labels, mask = policy.processor.preprocess(
        samples=[processor_sample],
        device=device,
        training=training,
        control_flag="context_only" if prefix else None,
    )

    valid = mask[0] > 0
    ids = ids[:, valid]
    labels = labels[:, valid]
    mask = mask[:, valid].float()

    if policy.model.proprio_embedder is not None:
        proprio = build_proprio_batch(
            [sample],
            device,
            torch.float32,
            zero_values=policy.model.proprio_encoder == "zeros",
        )
    else:
        proprio = None

    embeddings, grids = policy.model.encode_context_segment(
        ids,
        mask,
        pixels,
        proprio,
    )
    return ids, labels, mask, embeddings, grids, pixels


def append(
    policy: Any,
    state: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    embeddings: torch.Tensor | None = None,
    grids: Iterable[Any] = (),
    pixels: Any = None,
):
    """Append one segment transactionally and return a new InferenceState.

    The original state/cache is not mutated. Callers commit the returned state
    only after generation, ActionCodec decode and output validation succeed.
    """
    from .g05_policy import InferenceState

    mask = mask.float()
    if embeddings is None:
        embeddings, grids = policy.model.encode_context_segment(ids, mask)

    if state is not None:
        old_mask, old_grids = state.kv_cache.context_layout
        mask = torch.cat((old_mask, mask), dim=-1)
        grids = list(old_grids) + list(grids)
        full_ids = torch.cat((state.input_ids, ids), dim=-1)
    else:
        full_ids = ids

    _budget(policy, full_ids.shape[-1])
    hidden, cache, position_ids = policy.model.forward_context(
        embeddings,
        mask,
        grids,
        None if state is None else state.kv_cache,
    )

    output = InferenceState(
        kv_cache=cache,
        attention_mask=mask,
        position_ids=position_ids,
        last_hidden=hidden[:, -1],
        pixel_values=(
            pixels
            if pixels is not None
            else (state.pixel_values if state is not None else {})
        ),
        input_ids=full_ids,
        device=ids.device,
    )
    output.check_invariants("native append")
    if output.input_ids.shape != output.attention_mask.shape:
        raise AssertionError("committed input IDs and layout diverged")
    return output, hidden


def finalize_execution(policy: Any, state: Any, count: int | None):
    """Replace a proposal with the causal record of acknowledged commands.

    The original fixed-horizon ActionCodec may mix information across time, so
    its token stream must not be sliced. Re-encode only the applied continuous
    command prefix with deterministic hold padding to the original horizon.
    """
    from g05.data.native_episodes import hold_pad

    if state is None or state.pending_action is None:
        if count is not None:
            raise ValueError("execution receipt without pending action")
        return state

    sample = state.pending_action
    base = replace(state, pending_action=None, generated_ids=None)
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("pending action requires integer execution receipt")

    action_value = sample["action"]["value"]
    if not 0 <= count <= action_value.shape[0]:
        raise ValueError("execution receipt outside generated horizon")
    if count == 0:
        return base

    padded_action = hold_pad(action_value, count)
    sample = {
        **sample,
        "action": {**sample["action"], "value": padded_action},
    }
    ids, _, mask = policy.processor.preprocess(
        samples=[sample],
        device=policy.device,
        training=False,
        control_flag="return_suffix",
    )
    valid = mask[0] > 0
    ids = ids[0, valid].tolist()
    mask_values = mask[0, valid].tolist()
    if not ids:
        raise RuntimeError("empty executed action history")

    candidate = base
    for kind, group in groupby(
        zip(ids, mask_values),
        key=lambda item: item[1],
    ):
        group_items = list(group)
        group_ids = [token_id for token_id, _ in group_items]
        group_types = [kind] * len(group_items)
        candidate, _ = append_ids(policy, candidate, group_ids, group_types)
    return candidate


def remember_proposal(
    policy: Any,
    base: Any,
    candidate: Any,
    sample: dict[str, Any],
    action: torch.Tensor,
    action_dim_is_pad: torch.Tensor | None = None,
    parts_meta: Any = None,
):
    """Retain only what is needed to finalize this proposal after a receipt."""
    if not execution_history(policy):
        return candidate
    if base.pending_action is not None:
        raise RuntimeError("unconfirmed proposal must be finalized before generating")
    if action.ndim != 2 or not torch.isfinite(action).all():
        raise ValueError("invalid proposal")

    if action_dim_is_pad is None:
        padding = torch.zeros(
            action.shape[-1],
            dtype=torch.bool,
            device=action.device,
        )
    else:
        padding = torch.as_tensor(
            action_dim_is_pad,
            device=action.device,
            dtype=torch.bool,
        ).reshape(-1)
    if padding.numel() != action.shape[-1]:
        raise ValueError("invalid action padding mask")

    # Never read sample['action']['value']; it may contain a GT target.
    pending_sample = {
        **sample,
        "action": {
            "value": action.detach().clone().masked_fill(padding, 0),
            "action_dim_is_pad": padding,
            "action_op_mask": ~padding,
        },
    }
    if parts_meta is not None:
        pending_sample["action"]["parts_meta"] = parts_meta

    # Keep the pre-action committed cache. The speculative action cache is
    # discarded; generated IDs remain available for diagnostics only.
    return replace(
        base,
        pending_action=pending_sample,
        generated_ids=candidate.generated_ids,
        generated_texts=candidate.generated_texts,
    )


def append_observation(policy: Any, state: Any, samples: list[dict[str, Any]], pixels: Any):
    if len(samples) != 1:
        raise ValueError("native streams are isolated; use one sample per stream")
    ids, _, mask, embeddings, grids, processed_pixels = encode(
        policy,
        samples[0],
        pixels,
        prefix=True,
    )
    output, _ = append(
        policy,
        state,
        ids,
        mask,
        embeddings,
        grids,
        processed_pixels,
    )
    return output


def append_ids(policy: Any, state: Any, ids: list[int], types: list[float]):
    token_ids = torch.as_tensor(
        ids,
        device=policy.device,
        dtype=torch.long,
    ).reshape(1, -1)
    token_types = torch.as_tensor(
        types,
        device=policy.device,
        dtype=torch.float32,
    ).reshape(1, -1)
    if token_types.numel() == 1:
        token_types = token_types.expand_as(token_ids).clone()
    return append(policy, state, token_ids, token_types)


def commit(destination: Any, source: Any) -> None:
    """Commit a fully validated candidate InferenceState in place."""
    destination.__dict__.update(source.__dict__)


def text(
    policy: Any,
    state: Any,
    max_new_tokens: int | None = None,
    stop_token_ids: list[int] | None = None,
    trim_token_ids: list[int] | None = None,
    return_updated_state: bool = True,
    **kwargs: Any,
):
    """Generate text autoregressively while preserving the shared context state."""
    if execution_history(policy) and state.pending_action is not None:
        raise RuntimeError("finalize pending action before generating text")

    ar = policy.model.ar_helper
    token_budget = int(
        ar.max_new_tokens if max_new_tokens is None else max_new_tokens
    )
    if token_budget < 1:
        raise ValueError("positive text generation budget required")

    stop_ids = set(stop_token_ids or [])
    eos_token_id = policy.model.cfg.get("eos_token_id")
    if eos_token_id is not None:
        stop_ids.add(int(eos_token_id))

    candidate = state
    generated_tokens: list[int] = []
    do_sample = kwargs.get("do_sample", ar.do_sample)

    for _ in range(token_budget):
        logits = policy.model.vlm.decode(candidate.last_hidden).clone()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite text logits")

        # Never enter BAR during language generation; keep EOV and normal text controls.
        action_range = ar.action_token_range
        if action_range:
            logits[..., action_range[0] : action_range[1] + 1] = -torch.inf
        for token_id in (ar.bos_blk_id, ar.eos_blk_id):
            if token_id is not None:
                logits[..., token_id] = -torch.inf

        temperature = (
            kwargs.get("temperature", ar.temperature) if do_sample else 0
        )
        token = int(
            ar._sample(
                logits,
                temperature=temperature,
                top_k=kwargs.get("top_k", ar.top_k),
                top_p=kwargs.get("top_p", ar.top_p),
            ).item()
        )
        generated_tokens.append(token)
        candidate, _ = append_ids(
            policy,
            candidate,
            [token],
            [int(TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX)],
        )
        if token in stop_ids:
            break

    if stop_token_ids and generated_tokens[-1] not in set(stop_token_ids):
        raise RuntimeError(
            "required text stop token not generated; action stage not committed"
        )

    generated_ids = torch.tensor([generated_tokens], device=state.device)
    generated_texts = policy.processor.decode_text(
        generated_ids,
        trim_token_ids,
    )
    candidate = replace(
        candidate,
        generated_ids=generated_ids,
        generated_texts=generated_texts,
    )

    if return_updated_state:
        commit(state, candidate)
        return state
    return replace(
        state,
        generated_ids=generated_ids,
        generated_texts=generated_texts,
    )


def action_layout(policy: Any, sample: dict[str, Any]):
    """Build the exact original post-EOV BAR grammar with dummy action codes.

    The dummy values are used only to ask the original Processor for its static
    template/layout. They are never passed to the model or used as predictions.
    Only the currently verified fixed grouped serializer is accepted.
    """
    ar = policy.model.ar_helper
    bar_builder = policy.processor.bar_builder
    expected_tokens = ar._bar_expected_action_tokens

    valid_layout = (
        ar.block_wise_autoregressive
        and bar_builder.block_wise
        and ar.block_size == bar_builder.block_size
        and bar_builder.ar_block_num == 0
        and expected_tokens
        and expected_tokens % bar_builder.block_size == 0
        and len(ar._bar_position_constraints) == expected_tokens
    )
    if not valid_layout:
        raise ValueError(
            "native action generation requires fixed grouped BAR layout, ar_block_num=0"
        )

    real_token_length = int(policy.action_tokenizer.action_token_len_real)
    dummy_codes = [
        int(policy.action_tokenizer.action_token_begin_idx)
    ] * real_token_length
    sample_with_dummy_action = {**sample, "action": dummy_codes}

    processor = policy.processor
    segments = processor._prepare_segments(
        [sample_with_dummy_action],
        control_flag="return_suffix",
    )[0]
    action_segments = sum(
        segment.processor_key == "action" for segment in segments
    )
    has_dynamic_non_action = any(
        segment.type == "dynamic"
        and segment.sample_key
        and segment.processor_key != "action"
        for segment in segments
    )
    if action_segments != 1 or has_dynamic_non_action:
        raise ValueError(
            "native BAR suffix must contain exactly one action and static controls"
        )

    ids, _, mask, has_action = processor._preprocess_single_sample(
        segments,
        sample_with_dummy_action,
        processor.tokenizer,
        False,
        False,
    )
    if not has_action:
        raise ValueError("template has no post-EOV action segment")

    # BAR uses the native float mask discriminator for its action groups.
    groups = [
        (float(kind), list(group))
        for kind, group in groupby(
            zip(ids, mask),
            key=lambda pair: pair[1],
        )
    ]
    bar_blocks = [
        group
        for kind, group in groups
        if int(kind) == 3 and kind != 3
    ]
    expected_blocks = expected_tokens // bar_builder.block_size + 1
    if len(bar_blocks) != expected_blocks or any(
        len(group) != bar_builder.block_size for group in bar_blocks
    ):
        raise ValueError("unexpected native BAR template layout")
    if any(token_id != bar_builder.bos_blk_id for token_id, _ in bar_blocks[0]):
        raise ValueError("missing expanded BAR BOS")
    return groups


def action(
    policy: Any,
    state: Any,
    sample: dict[str, Any],
    max_new_tokens: int,
):
    """Generate a fixed-layout, grammar-constrained BAR action sequence.

    The original per-position and joint constraints are reused. Language
    repetition penalties are not applied to action codes. Return IDs contain
    only the generated action/control sequence needed by ``decode_ar``.
    """
    if execution_history(policy) and state.pending_action is not None:
        raise RuntimeError("pending proposal must be finalized before generating action")

    ar = policy.model.ar_helper
    bar_builder = policy.processor.bar_builder
    eov_token_id = getattr(policy.processor, "eov_token_id", None)
    if (
        eov_token_id is not None
        and int(state.input_ids[0, -1]) != int(eov_token_id)
    ):
        raise ValueError("action generation must start at the native EOV boundary")

    groups = action_layout(policy, sample)
    total_tokens = sum(len(group) for _, group in groups)
    if total_tokens > max_new_tokens:
        raise RuntimeError("native BAR token budget insufficient")
    _budget(policy, state.input_ids.size(-1) + total_tokens)

    candidate = state
    logits = None
    generated_codes: list[int] = []

    for kind, group in groups:
        group_ids = [token_id for token_id, _ in group]
        is_bar_group = int(kind) == 3 and kind != 3

        if is_bar_group and logits is not None:
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite action logits")
            logits = logits.clone()

            # Mask illegal IDs before sampling. Original joint constraints also
            # handle rule fields whose serialized positions cross block edges.
            for position in range(bar_builder.block_size):
                constraint = ar._bar_position_constraints[
                    len(generated_codes) + position
                ]
                if "fixed_token" in constraint:
                    fixed_token = constraint["fixed_token"]
                    fixed_logit = logits[0, position, fixed_token].clone()
                    logits[0, position].fill_(-torch.inf)
                    logits[0, position, fixed_token] = fixed_logit
                else:
                    lower = constraint["token_lower"]
                    upper = constraint["token_upper"]
                    logits[0, position, :lower] = -torch.inf
                    logits[0, position, upper:] = -torch.inf

            sampled = ar._sample(
                logits[0],
                temperature=ar.temperature if ar.do_sample else 0,
                top_k=ar.top_k,
                top_p=ar.top_p,
            )
            sampled = ar._enforce_bar_joint_constraints(
                logits,
                sampled,
                len(generated_codes),
                generated_codes,
            )
            group_ids = sampled.tolist()
            generated_codes.extend(group_ids)

        candidate, hidden = append_ids(
            policy,
            candidate,
            group_ids,
            [kind] * len(group_ids),
        )
        if is_bar_group:
            logits = policy.model.vlm.decode(hidden)

    if len(generated_codes) != ar._bar_expected_action_tokens:
        raise RuntimeError("incomplete action codes")

    # The full suffix is in the cache; this compact sequence is only for Codec decode.
    generated_ids = torch.tensor(
        [[bar_builder.bos_blk_id, *generated_codes, bar_builder.eos_blk_id]],
        device=state.device,
    )
    return candidate, generated_ids


def mask_fixed_action_targets(
    policy: Any,
    ids: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Optional CE ablation for deterministic grammar fields.

    This keeps the original model, LM head and CE. Only deterministic fields
    fixed by the original BAR grammar are masked; variable motion and rule
    digits remain supervised. The option is off by default.
    """
    ar = policy.model.ar_helper
    if not ar._bar_expected_action_tokens:
        raise ValueError("fixed-target mask requires fixed BAR grammar")

    spans = []
    cursor = 0
    for kind, group in groupby(mask[0].tolist()):
        length = sum(1 for _ in group)
        if int(kind) == 3 and kind != 3:
            spans.append((cursor, cursor + length))
        cursor += length

    if not spans:
        return labels

    width = ar.block_size
    expected_span_count = ar._bar_expected_action_tokens // width + 1
    if len(spans) != expected_span_count or any(
        end - start != width for start, end in spans
    ):
        raise ValueError("unexpected BAR spans in fixed-target ablation")

    masked_labels = labels.clone()
    for block_index, (previous_span, target_span) in enumerate(
        zip(spans, spans[1:])
    ):
        previous_start, _ = previous_span
        target_start, _ = target_span
        for position in range(width):
            constraint = ar._bar_position_constraints[block_index * width + position]
            if "fixed_token" not in constraint:
                continue

            fixed_token = constraint["fixed_token"]
            if (
                int(ids[0, target_start + position]) != fixed_token
                or int(masked_labels[0, previous_start + position + 1])
                not in (fixed_token, -100)
            ):
                raise ValueError("BAR grammar and training label disagree")
            masked_labels[0, previous_start + position + 1] = -100

    return masked_labels


def train(policy: Any, batch: dict[str, Any]):
    """Apply the original CE to complete same-Episode multimodal sequences.

    Each Episode is encoded as a B=1 sequence. In the default executed-history
    mode, historical turns are context-only and the current target uses original
    G05 labels. With ``native_replan_steps=0``, the collate path may retain labels
    on historical turns. Episode losses are weighted by valid supervised-token
    count to avoid overweighting short Episodes.
    """
    if policy.continuous_action or not policy.discrete_action:
        raise ValueError("native multi-turn training currently supports AR only")

    weighted_losses = []
    supervised_token_count = 0
    weighted_accuracy = 0.0
    action_token_count = 0
    action_correct = 0.0
    text_token_count = 0
    text_correct = 0.0

    for episode in batch["native_episodes"]:
        embeddings_list = []
        masks_list = []
        labels_list = []
        image_grids = []

        for turn in episode:
            sample = receipt(
                turn["samples"][0],
                turn.get("executed_steps"),
                executed_only=execution_history(policy),
            )
            ids, labels, mask, embeddings, grids, _ = encode(
                policy,
                sample,
                turn["pixel_values"],
                training=policy.training,
            )
            if not turn.get("supervise", True):
                labels = torch.full_like(labels, -100)
            elif policy.config.get("native_mask_fixed_action_targets", False):
                labels = mask_fixed_action_targets(policy, ids, labels, mask)

            embeddings_list.append(embeddings)
            masks_list.append(mask)
            labels_list.append(labels)
            image_grids.extend(grids)

        episode_labels = torch.cat(labels_list, dim=-1)
        valid_count = int((episode_labels[:, 1:] != -100).sum())
        if not valid_count:
            continue

        _budget(policy, episode_labels.size(-1))
        hidden, _, _ = policy.model.forward_context(
            torch.cat(embeddings_list, dim=1),
            torch.cat(masks_list, dim=1),
            image_grids,
        )
        loss, accuracy = policy.model.ar_helper.cal_ce_loss(
            hidden,
            episode_labels,
            policy.model,
        )

        weighted_losses.append(loss * valid_count)
        supervised_token_count += valid_count
        weighted_accuracy += float(accuracy) * valid_count

        ce_cache = policy.model.ar_helper._last_ce_cache
        shifted_labels = ce_cache["shift_labels_masked"]
        action_range = policy.model.ar_helper.action_token_range
        if action_range:
            action_mask = (
                (shifted_labels >= action_range[0])
                & (shifted_labels <= action_range[1])
            )
        else:
            action_mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
        text_mask = ~action_mask

        action_token_count += int(action_mask.sum())
        text_token_count += int(text_mask.sum())
        token_correct = ce_cache["_per_token_correct"]
        action_correct += float(token_correct[action_mask].sum())
        text_correct += float(token_correct[text_mask].sum())

    if not supervised_token_count:
        raise ValueError("episode batch contains no supervised tokens")

    loss = (
        sum(weighted_losses)
        / supervised_token_count
        * policy.language_loss_weight
    )
    policy.train_action_accuracy = weighted_accuracy / supervised_token_count
    policy.train_cot_accuracy = (
        text_correct / max(text_token_count, 1) if policy.predict_cot else 0.0
    )
    policy._train_acc.push("overall", weighted_accuracy / supervised_token_count)
    policy._train_acc.push("action_token", action_correct / max(action_token_count, 1))
    if policy.predict_cot:
        policy._train_acc.push("cot", policy.train_cot_accuracy)

    # Episode shapes differ from the legacy current-turn batch; do not expose a
    # misleading CE cache to callbacks that expect the legacy batch layout.
    policy.model.ar_helper._last_ce_cache = None
    policy._fwd_step += 1
    return loss, {"ce_loss": loss.detach()}
