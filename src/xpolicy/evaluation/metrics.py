"""Raw samples and explicit denominators; no average of trial percentiles."""

import numpy as np
import math


def process_cpu_seconds():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    except (ImportError, OSError, ValueError):
        return None


def series(values, *, unit="s", scope="local_monotonic"):
    values = list(values)
    if any(isinstance(value, (bool, str)) for value in values):
        raise ValueError("metric samples must be real numbers")
    values = [float(value) for value in values]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("metric samples must be finite and nonnegative")
    if not values:
        return {"status": "NOT_MEASURED", "samples": 0, "unit": unit, "scope": scope,
                "values": [], "p50": None, "p95": None, "p99": None, "max": None}
    # Percentile's direct interpolation can overflow at extreme finite values.
    scale = max(values)
    scaled = sorted(value / scale if scale else 0. for value in values)
    quantiles = []
    for percentage in (50, 95, 99):
        rank = (len(values) - 1) * percentage / 100.
        lo, hi = math.floor(rank), math.ceil(rank)
        fraction = rank - lo
        quantiles.append(scaled[lo] * (1 - fraction) + scaled[hi] * fraction)
    return {"status": "COMPUTED", "samples": len(values), "unit": unit, "scope": scope,
            "values": values, "p50": float(quantiles[0] * scale), "p95": float(quantiles[1] * scale),
            "p99": float(quantiles[2] * scale), "max": max(values)}
