"""XPolicyLab deployment loop for Native Context.

The environment is the source of truth for applied action steps and episode
completion. Replanning currently happens at action-chunk boundaries; this file
does not interrupt an in-flight command or fabricate measured trajectories.
"""

from __future__ import annotations

import os
from typing import Any

from XPolicyLab.policy.G05 import deploy as g05_deploy


EXECUTION_RECEIPT_KEY = "__knowact_execution__"


def _native_enabled() -> bool:
    value = os.environ.get("KNOWACT_NATIVE_CONTEXT", "true").lower()
    return value in {"true", "1"}


def _replan_steps() -> int:
    steps = int(os.environ.get("KNOWACT_REPLAN_STEPS", "16"))
    if steps < 1:
        raise ValueError("KNOWACT_REPLAN_STEPS must be positive")
    return steps


def _limit_action_chunk(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Defensively cap the action chunk at the configured execution interval."""
    return actions[:_replan_steps()]


def _attach_receipt(
    observation: dict[str, Any],
    receipt: dict[str, int] | None,
) -> dict[str, Any]:
    if receipt is None:
        return observation
    return {**observation, EXECUTION_RECEIPT_KEY: receipt}


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    """Evaluate one environment with per-chunk execution receipts."""
    if not _native_enabled():
        return g05_deploy.eval_one_episode(TASK_ENV, model_client)

    model_client.call(func_name="reset")
    plan_id = 0
    previous_receipt: dict[str, int] | None = None

    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(
            func_name="update_obs",
            obs=_attach_receipt(observation, previous_receipt),
        )
        action_chunk = _limit_action_chunk(model_client.call(func_name="get_action"))
        if not action_chunk:
            raise RuntimeError("KnowAct returned an empty action chunk")

        applied_steps = 0
        for action in action_chunk:
            TASK_ENV.take_action(action)
            applied_steps += 1
            if TASK_ENV.is_episode_end():
                break

        previous_receipt = {"plan": plan_id, "steps": applied_steps}
        plan_id += 1


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    """Evaluate a batch while keeping receipts isolated by environment key."""
    if not _native_enabled():
        return g05_deploy.eval_one_episode_batch(TASK_ENV, model_client)

    model_client.call(func_name="reset")
    receipts: dict[Any, dict[str, int]] = {}

    while not TASK_ENV.is_episode_end():
        env_keys = list(TASK_ENV.get_running_env_idx_list())
        observations = TASK_ENV.get_obs_batch(env_keys)
        if not env_keys or len(set(env_keys)) != len(env_keys):
            raise RuntimeError("invalid or duplicate environment keys")
        if len(observations) != len(env_keys):
            raise RuntimeError("observation batch does not match environment keys")

        model_client.call(
            func_name="update_obs_batch",
            obs=[
                _attach_receipt(observation, receipts.get(env_key))
                for env_key, observation in zip(env_keys, observations)
            ],
        )
        action_chunks = model_client.call(func_name="get_action_batch", obs=env_keys)
        if len(action_chunks) != len(env_keys):
            raise RuntimeError("action batch does not match environment keys")

        plans = {
            env_key: _limit_action_chunk(action_chunk)
            for env_key, action_chunk in zip(env_keys, action_chunks)
        }
        if any(not action_chunk for action_chunk in plans.values()):
            raise RuntimeError("KnowAct returned an empty action chunk")

        for env_key in env_keys:
            previous_plan = receipts.get(env_key, {}).get("plan", -1)
            receipts[env_key] = {"plan": previous_plan + 1, "steps": 0}

        max_chunk_length = max(len(action_chunk) for action_chunk in plans.values())
        for action_index in range(max_chunk_length):
            if TASK_ENV.is_episode_end():
                break

            running_keys = set(TASK_ENV.get_running_env_idx_list())
            active_keys = [
                env_key
                for env_key in env_keys
                if env_key in running_keys and action_index < len(plans[env_key])
            ]
            if not active_keys:
                break

            TASK_ENV.take_action_batch(
                [plans[env_key][action_index] for env_key in active_keys],
                active_keys,
            )
            for env_key in active_keys:
                receipts[env_key]["steps"] += 1


# XPolicyLab clients pass TASK_ENV by keyword or position.
