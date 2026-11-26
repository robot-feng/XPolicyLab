# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""PolicyInferencer — reusable batched inference for G05 policies."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from typing import Any

import torch

from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.data.data_utils import collate_fn_pad_sequences, custom_collate_fn
from g05.utils.common.pytorch_utils import dict_apply

logger = logging.getLogger(__name__)


def _sync_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclasses.dataclass(slots=True)
class PreparedSample:
    """Preprocessed sample plus its sub_processor reference."""

    obs_dict: dict[str, Any]
    sample: dict[str, Any]
    sub_processor: Any


def resolve_processor(processor, data):
    """Return the matching sub-processor from MixtureProcessor, or single directly.

    Supports both raw_obs (key="embodiment_type") and obs_dict (key="embodiment") forms.
    """
    if isinstance(processor, MixtureProcessor):
        emb = data.get("embodiment_type") or data.get("embodiment")
        if emb is None and len(processor.processors) == 1:
            emb = next(iter(processor.processors))
        return processor[emb]
    return processor


def action_padding_for_history(processor, action_dim):
    """Reuse the actual merger on shape-only zeros, never on target actions."""
    from g05.data.native_episodes import validate_history_processor
    validate_history_processor(processor)
    merger = getattr(processor, 'action_state_merger', None)
    if merger is None:
        return None  # already-flat/test processor
    raw = {m['key']: torch.zeros(1, int(m['shape'])) for m in processor.shape_meta['action']}
    pad = merger.forward({'action': raw})['action_dim_is_pad']
    if pad.shape != (action_dim,):
        raise ValueError('unexpected grouped action padding shape')
    return pad.bool()


class PolicyInferencer:
    """Unified inference wrapper: obs_dicts in → action dicts out."""

    def __init__(
        self,
        policy,
        processor,
        device: str = "cuda",
        action_source: str = "both",
    ):
        self.policy = policy
        self.processor = processor
        self.device = device
        if action_source not in {"both", "ar", "fm"}:
            raise ValueError(f"Unsupported inference action_source: {action_source!r}")
        self.action_source = action_source
        predict_params = inspect.signature(policy.predict_action).parameters.values()
        self._supports_action_source = any(
            param.name == "action_source" or param.kind == inspect.Parameter.VAR_KEYWORD
            for param in predict_params
        )
        if not self._supports_action_source:
            logger.warning(
                "Policy %s uses the legacy predict_action(batch) API; "
                "action_source=%s will be selected by the checkpoint configuration.",
                type(policy).__qualname__,
                action_source,
            )

    # ──────── Public API ────────

    def infer(self, obs_dicts: list[dict]) -> list[dict]:
        """Full pipeline: raw obs_dicts → postprocessed action dicts."""
        results, _ = self.infer_with_timing(obs_dicts)
        return results

    def infer_with_timing(self, obs_dicts: list[dict]) -> tuple[list[dict], dict[str, float]]:
        """Same as :meth:`infer` plus per-stage timing (ms).

        Returns ``(results, {"preprocess_ms", "infer_ms", "postprocess_ms"})``.
        """
        if not obs_dicts:
            return [], {"preprocess_ms": 0.0, "infer_ms": 0.0, "postprocess_ms": 0.0}

        _sync_if_cuda_available()
        t0 = time.monotonic()
        prepared = [self._prepare(obs_dict) for obs_dict in obs_dicts]
        padding_input_id = prepared[0].sub_processor.pad_token_id
        batch = self._collate([p.sample for p in prepared], padding_input_id=padding_input_id)
        _sync_if_cuda_available()
        t1 = time.monotonic()
        batch = self.forward_batch(batch)
        _sync_if_cuda_available()
        t2 = time.monotonic()
        model_timing = batch.pop("_timing", {})
        cot_texts = batch.get("cot_text")
        results = []
        for i, p in enumerate(prepared):
            action = self._postprocess_single(batch, index=i, sub_processor=p.sub_processor)
            if cot_texts is not None:
                action["_cot_text"] = cot_texts[i]
            results.append(action)
        _sync_if_cuda_available()
        t3 = time.monotonic()
        timing = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
        }
        if isinstance(model_timing, dict):
            timing.update(model_timing)
        return results, timing

    def infer_one(self, obs_dict: dict) -> dict:
        """Single obs convenience."""
        return self.infer([obs_dict])[0]

    def forward_batch(self, batch: dict) -> dict:
        """Core: to device → predict_action → to cpu."""
        batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self._supports_action_source:
                batch = self.policy.predict_action(batch, action_source=self.action_source)
            else:
                batch = self.policy.predict_action(batch)
        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        return batch

    # ──────── Private helpers ────────

    def _prepare(self, obs_dict: dict) -> PreparedSample:
        """Preprocess one request and annotate its batching key."""
        sub_processor = resolve_processor(self.processor, obs_dict)
        sample = sub_processor.preprocess(obs_dict)

        sample.pop("action", None)
        sample.pop("action_is_pad", None)
        sample.pop("gt_action", None)

        return PreparedSample(
            obs_dict=obs_dict,
            sample=sample,
            sub_processor=sub_processor,
        )

    @staticmethod
    def _collate(samples: list[dict], padding_input_id: int) -> dict:
        """Collate inference samples, padding text fields when present."""
        batch = [dict(sample) for sample in samples]
        first = batch[0]
        has_text_tokens = {"input_ids", "labels", "attention_mask"} <= set(first.keys())
        if has_text_tokens:
            return collate_fn_pad_sequences(batch, padding_input_id=padding_input_id)

        collated = custom_collate_fn(batch)
        if "samples" in first:
            collated["samples"] = [sample["samples"] for sample in batch]
        sample_meta_keys = ("idx", "task", "embodiment", "dataset_locator", "frequency")
        collated["sample_meta"] = [
            {key: sample.get(key) for key in sample_meta_keys if key in sample} for sample in batch
        ]
        return collated

    @staticmethod
    def _postprocess_single(batch: dict, index: int, sub_processor) -> dict:
        """Slice one sample from a batched model output and postprocess it."""
        item_batch = {
            "action": batch["action"][index : index + 1],
            "proprio": batch["proprio"][index : index + 1],
        }
        if "action_dim_is_pad" in batch:
            item_batch["action_dim_is_pad"] = batch["action_dim_is_pad"][index : index + 1]
        if "proprio_dim_is_pad" in batch:
            item_batch["proprio_dim_is_pad"] = batch["proprio_dim_is_pad"][index : index + 1]
        if "action_op_mask" in batch:
            item_batch["action_op_mask"] = batch["action_op_mask"][index : index + 1]
        action = sub_processor.postprocess(item_batch)["action"]
        ar_absent_keys = batch.get("ar_absent_keys")
        absent = (
            ar_absent_keys[index]
            if ar_absent_keys is not None and index < len(ar_absent_keys)
            else set()
        )
        if absent:
            action["_absent_keys"] = absent
        return action

    def infer_stream(self, obs_dict, state=None, *, executed_steps=None, demonstration=()):
        """Native single-stream action; reuse all original preprocess/postprocess.

        No cache is published to caller until decoding AND postprocessing succeed.
        Caller owns the per-environment state and actual execution receipts.
        """
        prepared = self._prepare(obs_dict)
        batch = self._collate([prepared.sample], prepared.sub_processor.pad_token_id)
        if int(self.policy.config.get('native_replan_steps', 0)) > 0:
            pad = action_padding_for_history(prepared.sub_processor, int(self.policy.model_config.action_dim))
            if pad is not None:
                if 'action_dim_is_pad' in batch and not torch.equal(batch['action_dim_is_pad'].reshape(-1).cpu(), pad):
                    raise ValueError('observed action padding disagrees with robot layout')
                batch['action_dim_is_pad'] = pad[None]
        batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
        self.policy.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=str(self.device).startswith('cuda')):
            if demonstration:
                from g05.data.native_episodes import visual_turn

                if state is not None or executed_steps is not None:
                    raise ValueError("A demonstration must precede the first execution turn")
                with_proprio = self.policy.config.get("native_demo_proprio", False)
                for frame in demonstration:
                    # Earlier image-only callers passed the camera mapping directly.
                    turn = visual_turn(prepared.sub_processor, frame.get("images", frame),
                                       batch["samples"][0], frame["state"] if with_proprio else None)
                    state = self.policy.append_context(state, turn["samples"], turn["pixel_values"])
            candidate = self.policy.append_context(state, batch['samples'], batch['pixel_values'],
                                                   executed_steps=executed_steps)
            if self.policy.predict_cot:
                stop = self.policy.model.ar_helper.eov_token_id
                if stop is None:
                    raise ValueError('CoT requires original EOV token')
                self.policy.generate_text(candidate, stop_token_ids=[stop],
                                          trim_token_ids=self.policy._get_trim_token_ids(True))
            result = self.policy.generate_action(candidate, batch['samples'],
                         action_dim_is_pad=batch.get('action_dim_is_pad'), action_source='ar')
        batch.update(result)
        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        action = self._postprocess_single(batch, 0, prepared.sub_processor)
        if self.policy.predict_cot and candidate.generated_texts:
            action['_cot_text'] = candidate.generated_texts[0]
        return action, candidate, result.get('_timing', {})
