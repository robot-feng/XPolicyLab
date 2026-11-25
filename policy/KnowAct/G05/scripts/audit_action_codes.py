"""Audit BAR dependence using paired expert alternatives encoded by a real codec.

Input is an integer NumPy array with shape [N, 2, L]. Each pair must contain two
valid alternative action-code sequences for the same available context. This is
a controlled binary-mixture diagnostic, not a policy success metric and not an
empirical independence estimate from arbitrary unpaired demonstrations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def paired_audit(pairs: np.ndarray, block_width: int) -> dict[str, Any]:
    """Estimate how often independent per-position sampling preserves a pair.

    For a pair whose first differing BAR block contains ``d`` differing
    positions, the equal-weight binary-mixture diagnostic reports ``2**(1-d)``.
    This assumes two equally likely modes and independent marginal sampling.
    """
    pairs = np.asarray(pairs)
    valid_shape = (
        pairs.ndim == 3
        and pairs.shape[1] == 2
        and pairs.shape[0] >= 1
        and pairs.shape[2] >= 1
        and np.issubdtype(pairs.dtype, np.integer)
        and block_width >= 1
    )
    if not valid_shape:
        raise ValueError(
            "expected nonempty integer [contexts, 2, L] and positive block width"
        )

    rows: list[dict[str, Any]] = []
    for first, second in pairs:
        differing_positions = np.flatnonzero(first != second)
        if len(differing_positions) == 0:
            rows.append(
                {
                    "collision": True,
                    "first_block": None,
                    "differing_positions": 0,
                    "independent_mode_mass": 1.0,
                }
            )
            continue

        first_block = int(differing_positions[0] // block_width)
        differing_in_first_block = int(
            np.count_nonzero((differing_positions // block_width) == first_block)
        )
        rows.append(
            {
                "collision": False,
                "first_block": first_block,
                "differing_positions": differing_in_first_block,
                "independent_mode_mass": 2.0 ** (1 - differing_in_first_block),
            }
        )

    usable = [row for row in rows if not row["collision"]]
    return {
        "block_width": block_width,
        "contexts": len(rows),
        "codec_collisions": len(rows) - len(usable),
        "distinct_contexts": len(usable),
        "mean_independent_mode_mass": (
            float(np.mean([row["independent_mode_mass"] for row in usable]))
            if usable
            else None
        ),
        "mean_differing_positions": (
            float(np.mean([row["differing_positions"] for row in usable]))
            if usable
            else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", required=True, help="Paired [N, 2, L] integer .npy")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    code_pairs = np.load(args.codes, allow_pickle=False)
    results = [paired_audit(code_pairs, width) for width in (1, 8, 16, 32)]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
