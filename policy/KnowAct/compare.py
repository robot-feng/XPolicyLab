"""Build paired XPolicyLab evaluations and score paired episode outcomes.

The launcher is dry-run by default. G05 and KnowAct run in separate policy
processes/environments because both use the vendored ``g05`` import namespace.
This module does not parse simulator logs or invent episode outcomes; provide a
paired CSV to ``--score`` after exporting the benchmark's actual success flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any


XPL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G05_EXECUTION_STEPS = 16
DEFAULT_REPLAN_STEPS = 16
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 17
SCORE_COLUMNS = {"task", "seed", "episode_id", "model", "success"}


def _get(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    """Read an argument with a default, also supporting lightweight test fixtures."""
    return getattr(args, name, default)


def _model_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the requested model variants with their own checkpoint semantics."""
    variants: list[dict[str, Any]] = [
        {
            "model": "G05",
            "policy": "G05",
            "checkpoint": args.g05_ckpt,
            "action_source": _get(args, "g05_action_source", "fm"),
            "observation_offsets": _get(args, "baseline_offsets", "0"),
            "keep_history": None,
        },
        {
            "model": "KnowAct",
            "policy": "KnowAct",
            "checkpoint": args.knowact_ckpt,
            "action_source": "ar",
            "observation_offsets": None,
            "keep_history": True,
        },
    ]

    if _get(args, "reset_ablation", False):
        variants.append(
            {
                "model": "KnowAct_reset",
                "policy": "KnowAct",
                "checkpoint": args.knowact_ckpt,
                "action_source": "ar",
                "observation_offsets": None,
                "keep_history": False,
            }
        )

    memory_checkpoint = _get(args, "g05_memory_ckpt")
    if memory_checkpoint:
        variants.append(
            {
                "model": "G05_MEM",
                "policy": "G05",
                "checkpoint": memory_checkpoint,
                "action_source": _get(
                    args,
                    "g05_memory_action_source",
                    _get(args, "g05_action_source", "fm"),
                ),
                "observation_offsets": _get(args, "g05_memory_offsets"),
                "keep_history": None,
            }
        )

    current_checkpoint = _get(args, "current_only_ckpt")
    if current_checkpoint:
        variants.append(
            {
                "model": "KnowAct_current",
                "policy": "KnowAct",
                "checkpoint": current_checkpoint,
                "action_source": "ar",
                "observation_offsets": None,
                "keep_history": False,
            }
        )

    return variants


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create deterministic per-task/per-seed commands and environment settings."""
    g05_steps = int(_get(args, "g05_execution_steps", DEFAULT_G05_EXECUTION_STEPS))
    knowact_steps = int(_get(args, "replan_steps", DEFAULT_REPLAN_STEPS))
    matched_interval = g05_steps == knowact_steps
    allow_unmatched = bool(_get(args, "allow_unmatched_interval", False))

    if not 1 <= g05_steps <= 32:
        raise ValueError("G05 execution interval must be in 1..32")
    if not 1 <= knowact_steps <= 32:
        raise ValueError("KnowAct replan interval must be in 1..32")
    if not matched_interval and not allow_unmatched:
        raise ValueError(
            "G05 and KnowAct execution intervals differ; use --allow-unmatched-interval "
            "only for a labeled stress test"
        )

    rows: list[dict[str, Any]] = []
    for task in args.task:
        for seed in args.seed:
            for variant in _model_variants(args):
                model_name = variant["model"]
                is_g05 = variant["policy"] == "G05"
                checkpoint = str(Path(variant["checkpoint"]).expanduser().resolve())

                env: dict[str, str] = {}
                if is_g05:
                    env.update(
                        {
                            "G05_CKPT_PATH": checkpoint,
                            "ROBODOJO_G05_ACTION_SOURCE": variant["action_source"],
                            "ROBODOJO_G05_OBS_OFFSETS": str(
                                variant["observation_offsets"]
                            ),
                        }
                    )
                    execution_steps = g05_steps
                    policy_env = args.g05_env
                    eval_script = XPL_ROOT / "policy" / "G05" / "eval.sh"
                else:
                    env.update(
                        {
                            "KNOWACT_CKPT_PATH": checkpoint,
                            "KNOWACT_NATIVE_CONTEXT": "true",
                            "KNOWACT_KEEP_HISTORY": str(
                                bool(variant["keep_history"])
                            ).lower(),
                            "KNOWACT_REPLAN_STEPS": str(knowact_steps),
                            "ROBODOJO_KNOWACT_ACTION_SOURCE": "ar",
                            "KNOWACT_ALLOW_STRIDE_SHIFT": str(
                                not matched_interval
                            ).lower(),
                        }
                    )
                    execution_steps = knowact_steps
                    policy_env = args.knowact_env
                    eval_script = XPL_ROOT / "policy" / "KnowAct" / "eval.sh"

                command = [
                    "bash",
                    str(eval_script),
                    args.bench,
                    task,
                    model_name,
                    args.env_cfg,
                    "joint",
                    str(seed),
                    args.policy_gpu,
                    args.env_gpu,
                    policy_env,
                    args.eval_env,
                ]
                rows.append(
                    {
                        "model": model_name,
                        "task": task,
                        "seed": int(seed),
                        "checkpoint": checkpoint,
                        "action_source": variant["action_source"],
                        "observation_offsets": variant["observation_offsets"],
                        "keep_history": variant["keep_history"],
                        "execution_steps": execution_steps,
                        "matched_execution_interval": matched_interval,
                        "environment": env,
                        "command": command,
                    }
                )
    return rows


def _load_paired_scores(csv_path: str | Path) -> dict[str, dict[tuple[str, str, str], int]]:
    """Load strict binary outcomes keyed by (task, seed, episode_id)."""
    by_model: dict[str, dict[tuple[str, str, str], int]] = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not SCORE_COLUMNS.issubset(reader.fieldnames):
            missing = SCORE_COLUMNS - set(reader.fieldnames or [])
            raise ValueError(f"score CSV missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            model = (row.get("model") or "").strip()
            task = (row.get("task") or "").strip()
            seed = (row.get("seed") or "").strip()
            episode_id = (row.get("episode_id") or "").strip()
            if not all((model, task, seed, episode_id)):
                raise ValueError(f"empty key field in score CSV line {line_number}")
            try:
                success = int(row["success"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-integer success value at line {line_number}") from exc
            if success not in (0, 1):
                raise ValueError(f"success must be 0 or 1 at line {line_number}")

            key = (task, seed, episode_id)
            model_rows = by_model.setdefault(model, {})
            if key in model_rows:
                raise ValueError(f"duplicate episode for model={model}, key={key}")
            model_rows[key] = success

    return by_model


def score_paired_episodes(
    csv_path: str | Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score paired outcomes and cluster-bootstrap task/seed groups.

    The interval is descriptive when the number of independent task/seed
    clusters is small; it is not a substitute for a pre-registered statistical
    analysis with enough independent tasks and seeds.
    """
    by_model = _load_paired_scores(csv_path)
    if "G05" not in by_model or "KnowAct" not in by_model:
        raise ValueError("score CSV must contain both G05 and KnowAct")

    baseline = by_model["G05"]
    if not baseline:
        raise ValueError("score CSV contains no episode outcomes")
    baseline_keys = set(baseline)
    results: dict[str, Any] = {}

    for model, outcomes in by_model.items():
        if set(outcomes) != baseline_keys:
            missing = len(baseline_keys - set(outcomes))
            extra = len(set(outcomes) - baseline_keys)
            raise ValueError(
                f"unpaired episodes for {model}: missing={missing}, extra={extra}"
            )

        keys = sorted(baseline_keys)
        differences = [outcomes[key] - baseline[key] for key in keys]
        cluster_differences: dict[tuple[str, str], list[int]] = {}
        for key, difference in zip(keys, differences):
            cluster_differences.setdefault(key[:2], []).append(difference)

        clusters = list(cluster_differences.values())
        confidence_interval = None
        if len(clusters) >= 2 and bootstrap_replicates > 0:
            rng = random.Random(bootstrap_seed)
            estimates = []
            for _ in range(bootstrap_replicates):
                sampled = [rng.choice(clusters) for _ in clusters]
                total = sum(sum(cluster) for cluster in sampled)
                count = sum(len(cluster) for cluster in sampled)
                estimates.append(total / count)
            estimates.sort()
            low = math.floor(0.025 * (len(estimates) - 1))
            high = math.ceil(0.975 * (len(estimates) - 1))
            confidence_interval = [estimates[low], estimates[high]]

        results[model] = {
            "episodes": len(outcomes),
            "success_rate": sum(outcomes.values()) / len(outcomes),
            "paired_delta_vs_G05": sum(differences) / len(differences),
            "task_seed_clusters": len(clusters),
            "cluster_bootstrap_95_interval": confidence_interval,
            "bootstrap_replicates": bootstrap_replicates if confidence_interval else 0,
        }

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired XPolicyLab benchmark launcher (dry-run by default)."
    )
    parser.add_argument("--score", help="Score a paired episode CSV instead of launching runs")
    parser.add_argument("--task", action="append", help="Benchmark task; repeatable")
    parser.add_argument("--seed", action="append", type=int, help="Evaluation seed; repeatable")
    parser.add_argument("--g05-ckpt")
    parser.add_argument("--knowact-ckpt")
    parser.add_argument("--g05-action-source", choices=("auto", "fm", "ar", "both"), default="fm")
    parser.add_argument(
        "--g05-memory-ckpt",
        help="Optional G05-MEM checkpoint, evaluated through the original G05 adapter",
    )
    parser.add_argument(
        "--g05-memory-action-source",
        choices=("auto", "fm", "ar", "both"),
        help="Defaults to --g05-action-source",
    )
    parser.add_argument("--g05-memory-offsets", help="Required observation offsets for G05-MEM")
    parser.add_argument("--current-only-ckpt", help="Optional separately trained single-turn KnowAct")
    parser.add_argument("--g05-env", default="base")
    parser.add_argument("--knowact-env", default="base")
    parser.add_argument("--eval-env", default="base")
    parser.add_argument("--bench", default="RoboDojo")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--policy-gpu", default="0")
    parser.add_argument("--env-gpu", default="0")
    parser.add_argument("--baseline-offsets", default="0")
    parser.add_argument("--g05-execution-steps", type=int, default=DEFAULT_G05_EXECUTION_STEPS)
    parser.add_argument("--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS)
    parser.add_argument("--output", default="comparison_runs")
    parser.add_argument("--reset-ablation", action="store_true")
    parser.add_argument("--allow-unmatched-interval", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually launch; default is dry-run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.score:
        print(json.dumps(score_paired_episodes(args.score), indent=2))
        return

    required = (args.task, args.seed, args.g05_ckpt, args.knowact_ckpt)
    if not all(required):
        parser.error("--task, --seed, --g05-ckpt and --knowact-ckpt are required")
    if any(not task or "/" in task or "\\" in task or task in (".", "..") for task in args.task):
        parser.error("--task must be a benchmark task name, not a path")
    if args.g05_memory_ckpt and not args.g05_memory_offsets:
        parser.error("--g05-memory-ckpt requires --g05-memory-offsets")
    if len(set(args.task)) != len(args.task) or len(set(args.seed)) != len(args.seed):
        parser.error("duplicate task or evaluation seed")

    try:
        rows = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": {
            "benchmark": args.bench,
            "environment_config": args.env_cfg,
            "g05_action_source": args.g05_action_source,
            "g05_execution_steps": args.g05_execution_steps,
            "knowact_action_source": "ar",
            "knowact_replan_steps": args.replan_steps,
            "matched_execution_interval": args.g05_execution_steps == args.replan_steps,
            "note": "Dry-run manifest unless --run is supplied. Episode outcomes are not inferred here.",
        },
        "runs": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in rows:
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in row["environment"].items()
        )
        print(assignments, shlex.join(row["command"]))

    if not args.run:
        print(f"Dry-run only. Inspect {manifest_path}; add --run to launch evaluations.")
        return

    for row in rows:
        if not Path(row["checkpoint"]).exists():
            raise FileNotFoundError(f"checkpoint not found: {row['checkpoint']}")
        eval_script = Path(row["command"][1])
        if not eval_script.is_file():
            raise FileNotFoundError(f"XPolicyLab eval script not found: {eval_script}")

    for index, row in enumerate(rows):
        log_path = output_dir / (
            f"{index:04d}_{row['model']}_{row['task']}_seed{row['seed']}.log"
        )
        environment = os.environ.copy()
        environment.update(row["environment"])
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                row["command"],
                env=environment,
                cwd=XPL_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row["returncode"] = completed.returncode
        row["log"] = str(log_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"evaluation failed with code {completed.returncode}; see {log_path}")


if __name__ == "__main__":
    main()
