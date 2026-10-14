"""Explicit state precision at the raw and normalized model boundaries."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping


def state_dtype(value: str, *, normalized: bool = False) -> np.dtype:
    allowed = ("float16", "float32") if normalized else ("float32", "float64")
    if not isinstance(value, str) or value not in allowed:
        key = "model_state_dtype" if normalized else "state_input_dtype"
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return np.dtype(value)


def cast_state(value, dtype: str, *, normalized: bool = False) -> np.ndarray:
    """Return an owned, finite real array without silently accepting overflow."""
    target = state_dtype(dtype, normalized=normalized)
    array = np.asarray(value)
    if array.ndim not in (1, 2) or not array.size or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("state must be a nonempty finite real (D,) or (T, D) array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=target, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"state is not representable as {target.name}")
    return result


def resolve_state_precision(config: Mapping, environment: Mapping) -> tuple[str, str]:
    """Explicit environment overrides, then deploy config, then legacy defaults."""
    if not isinstance(config, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("state precision configuration and environment must be mappings")
    raw = environment.get("STARVLA_STATE_INPUT_DTYPE") or config.get("state_input_dtype", "float32")
    packed = environment.get("STARVLA_MODEL_STATE_DTYPE") or config.get("model_state_dtype", "float32")
    state_dtype(raw)
    state_dtype(packed, normalized=True)
    return raw, packed
