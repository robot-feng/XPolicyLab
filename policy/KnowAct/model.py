"""Thin XPolicyLab adapter around the KnowAct-maintained G0.5 source.

Observation conversion, action postprocessing and checkpoint conventions are
inherited from the original G05 adapter. Native Context keeps one committed
InferenceState per environment. Run G05 and KnowAct in separate policy-server
processes / Python environments because both import the top-level ``g05`` name.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.policy.G05.model import BATCH_ENV_INDICES_KEY, BATCH_OBSERVATIONS_KEY, _step_sort_key
from XPolicyLab.policy.G05.model import Model as G05Adapter
from XPolicyLab.policy.KnowAct.demonstration import frame_indices
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root


POLICY_DIR = Path(__file__).resolve().parent


class Model(G05Adapter):
    """XPolicyLab Policy Adapter with per-environment Native Context state."""

    def __init__(self, model_cfg: Any):
        config = dict(model_cfg)
        if config.get("env_cfg_type") == "panda":
            config["eval_embodiment"] = config.get("eval_embodiment") or "panda"
            config["frequency"] = config.get("frequency") or 20
        g05_root = Path(
            config.get("g05_root")
            or os.environ.get("KNOWACT_ROOT")
            or POLICY_DIR / "G05"
        ).expanduser().resolve()

        self._assert_isolated_g05_imports(g05_root)
        config["g05_root"] = str(g05_root)

        self.native_context = str(
            os.environ.get(
                "KNOWACT_NATIVE_CONTEXT",
                config.get("native_context", True),
            )
        ).lower() in {"true", "1"}

        config["action_source"] = (
            os.environ.get("ROBODOJO_KNOWACT_ACTION_SOURCE")
            or config.get("action_source")
            or "ar"
        )
        if self.native_context:
            if config["action_source"] not in {"auto", "ar"}:
                raise ValueError("Native KnowAct currently supports AR only")
            config["action_source"] = "ar"

        overrides = [str(item) for item in config.get("hydra_overrides", [])]
        overrides = [
            item
            for item in overrides
            if not item.lstrip("+").startswith("model.model_arch.native_context=")
        ]
        # Do not silently alter BAR width or tokenizer layout of an existing checkpoint.
        overrides.append(
            f"model.model_arch.native_context={str(self.native_context).lower()}"
        )
        if token_limit := os.environ.get("KNOWACT_CONTEXT_MAX_TOKENS"):
            token_limit = int(token_limit)
            if token_limit <= 0:
                raise ValueError("KNOWACT_CONTEXT_MAX_TOKENS must be positive")
            overrides.append(f"model.model_arch.native_context_max_tokens={token_limit}")
        config["hydra_overrides"] = overrides

        super().__init__(config)

        self.keep_history = os.environ.get("KNOWACT_KEEP_HISTORY", "true").lower() in {
            "true",
            "1",
        }
        self._native_states: dict[Any, tuple[Any, int, int]] = {}
        self._demonstrations = {}
        self._native_lock = threading.RLock()

        if self.native_context:
            self._validate_native_runtime()

    @staticmethod
    def _assert_isolated_g05_imports(g05_root: Path) -> None:
        """Reject a process that already imported G05 from another checkout."""
        for module_name in ("g05", "scripts.serve_policy"):
            module = sys.modules.get(module_name)
            if module is None:
                continue

            module_paths = list(getattr(module, "__path__", ()))
            module_file = getattr(module, "__file__", None)
            if module_file:
                module_paths.append(module_file)

            imported_elsewhere = any(
                not Path(module_path).resolve().is_relative_to(g05_root)
                for module_path in module_paths
            )
            if imported_elsewhere:
                raise RuntimeError(
                    "G05 and KnowAct require separate policy-server processes"
                )

    def _validate_native_runtime(self) -> None:
        """Validate checkpoint layout and training/deployment stride agreement."""
        action_helper = self.policy.model.ar_helper
        if (
            not action_helper.block_wise_autoregressive
            or action_helper.block_size != 16
        ):
            raise ValueError(
                "Native KnowAct requires a BAR16-trained checkpoint; "
                "do not change its width at inference"
            )

        trained_stride = int(self.policy.config.get("native_replan_steps", 0))
        dispatch_stride = int(os.environ.get("KNOWACT_REPLAN_STEPS", "16"))
        if not 1 <= dispatch_stride <= self.policy.model_config.horizon_steps:
            raise ValueError("invalid deployment replan interval")

        allow_stride_shift = os.environ.get(
            "KNOWACT_ALLOW_STRIDE_SHIFT", "false"
        ).lower() in {"1", "true"}
        if trained_stride and trained_stride != dispatch_stride and not allow_stride_shift:
            raise ValueError("training/deployment replan interval mismatch")

        self.action_steps = dispatch_stride
        if getattr(self.processor, "num_obs_steps", 1) != 1:
            raise ValueError(
                "Native Context requires current-frame processor, not legacy MEM fusion"
            )

    def reset(self) -> None:
        with self._native_lock:
            super().reset()
            self._native_states.clear()
            self._demonstrations = {}

    def _remember_demo(self, obs, key):
        marker = obs.get("demonstration_frame")
        if marker is None or not self.native_context:
            return
        limit = int(self.policy.config.get("native_demo_frames", 0))
        if not limit:
            return
        index, length = marker["index"], marker["length"]
        if not 0 <= index < length or key in self._native_states:
            raise ValueError("Unexpected demonstration frame outside the episode prefix")
        if index in frame_indices(length, limit):
            self._demonstrations.setdefault(key, {})[index] = obs

    def update_obs(self, obs):
        with self._native_lock:
            if BATCH_OBSERVATIONS_KEY in obs:
                keys = obs.get(BATCH_ENV_INDICES_KEY)
                for i, frame in enumerate(obs[BATCH_OBSERVATIONS_KEY]):
                    self._remember_demo(frame, keys[i] if keys is not None else self._env_key_from_obs(frame, i))
            else:
                self._remember_demo(obs, self._env_key_from_obs(obs, 0))
            super().update_obs(obs)

    def update_obs_batch(self, obs_list):
        with self._native_lock:
            for i, obs in enumerate(obs_list):
                self._remember_demo(obs, self._env_key_from_obs(obs, i))
            super().update_obs_batch(obs_list)

    def _predict_chunks(self, observations: list[dict[str, Any]], env_keys=None):
        if not self.native_context:
            return super()._predict_chunks(observations, env_keys)

        if env_keys is None:
            env_keys = [
                self._env_key_from_obs(observation, fallback_idx=index)
                for index, observation in enumerate(observations)
            ]
        if len(env_keys) != len(observations) or len(set(env_keys)) != len(env_keys):
            raise ValueError("Native batch requires unique stable environment IDs")

        # Sequential per-environment compute is the correctness baseline. Publish
        # the entire batch atomically, not a partially successful batch.
        with self._native_lock:
            updates = {}
            action_chunks = []

            for env_key, observation in zip(env_keys, observations):
                previous = self._native_states.get(env_key)
                receipt = observation.get("__knowact_execution__")

                if previous is None:
                    if receipt is not None:
                        raise RuntimeError("receipt without a pending plan")
                    state, executed_steps, next_plan_id = None, None, 0
                else:
                    state, previous_plan_id, sent_steps = previous
                    if (
                        not isinstance(receipt, dict)
                        or receipt.get("plan") != previous_plan_id
                    ):
                        raise RuntimeError(
                            "missing, duplicated or stale execution receipt"
                        )
                    executed_steps = receipt.get("steps")
                    if (
                        not isinstance(executed_steps, int)
                        or isinstance(executed_steps, bool)
                        or not 0 <= executed_steps <= sent_steps
                    ):
                        raise ValueError("execution receipt exceeds dispatched steps")
                    next_plan_id = previous_plan_id + 1

                if not self.keep_history:
                    state, executed_steps = None, None

                raw_observation = self._to_g05_raw_obs(observation)
                observation_dict = self._build_obs_dict(raw_observation, self.processor)
                demo = self._demonstrations.get(env_key, {}) if state is None else {}
                if demo:
                    length = next(iter(demo.values()))["demonstration_frame"]["length"]
                    expected = frame_indices(length, self.policy.config["native_demo_frames"])
                    if sorted(demo) != expected:
                        raise ValueError("Incomplete demonstration before the first action")
                action, candidate_state, _ = self.inferencer.infer_stream(
                    observation_dict,
                    state,
                    executed_steps=executed_steps,
                    demonstration=[
                        self._build_obs_dict(self._to_g05_raw_obs(demo[i]), self.processor)["images"]
                        for i in sorted(demo)
                    ],
                )
                chunk = self._format_action_chunk(action)
                invalid_action = not chunk or any(
                    not np.isfinite(np.asarray(value)).all()
                    for step in chunk
                    for value in step.values()
                )
                if invalid_action:
                    raise RuntimeError("invalid postprocessed action; state not committed")

                updates[env_key] = (candidate_state, next_plan_id, len(chunk))
                action_chunks.append(chunk)

            self._native_states.update(updates)
            return action_chunks

    def _resolve_ckpt_path(self, model_cfg: Any) -> Path:
        config = dict(model_cfg)
        if not config.get("ckpt_path") and os.environ.get("KNOWACT_CKPT_PATH"):
            config["ckpt_path"] = os.environ["KNOWACT_CKPT_PATH"]

        checkpoint_root = resolve_checkpoint_root(
            config,
            POLICY_DIR / "checkpoints",
            policy_dir=POLICY_DIR,
        )
        if checkpoint_root.is_file():
            return checkpoint_root

        candidates = [checkpoint_root / "last.pt", checkpoint_root / "model_state_dict.pt"]
        candidates.extend(sorted(checkpoint_root.rglob("step_*.pt"), key=_step_sort_key, reverse=True))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"No G0.5-format .pt checkpoint found under {checkpoint_root}")
