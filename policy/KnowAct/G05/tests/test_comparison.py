"""Test paired benchmark planning and statistics without running a simulator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


COMPARE_PATH = Path(__file__).resolve().parents[2] / "compare.py"
SPEC = importlib.util.spec_from_file_location("knowact_compare", COMPARE_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _args(**overrides):
    values = {
        "task": ["task_a"],
        "seed": [0],
        "reset_ablation": False,
        "g05_ckpt": "/checkpoints/g05",
        "knowact_ckpt": "/checkpoints/knowact",
        "g05_action_source": "fm",
        "baseline_offsets": "0",
        "replan_steps": 16,
        "g05_execution_steps": 16,
        "allow_unmatched_interval": False,
        "bench": "RoboDojo",
        "env_cfg": "arx_x5",
        "policy_gpu": "0",
        "env_gpu": "1",
        "g05_env": "g05-env",
        "knowact_env": "knowact-env",
        "eval_env": "sim-env",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ComparisonTest(unittest.TestCase):
    def test_paired_manifest_uses_fm_for_official_g05_default(self):
        args = _args(
            task=["task_a", "task_b"],
            seed=[1, 2, 3],
            reset_ablation=True,
        )
        rows = compare.build_plan(args)
        self.assertEqual(len(rows), 18)
        for index in range(0, len(rows), 3):
            baseline, knowact, reset = rows[index : index + 3]
            self.assertEqual((baseline["task"], baseline["seed"]), (knowact["task"], knowact["seed"]))
            self.assertEqual(knowact["checkpoint"], reset["checkpoint"])
            self.assertTrue(baseline["command"][1].endswith("/policy/G05/eval.sh"))
            self.assertEqual(baseline["environment"]["ROBODOJO_G05_ACTION_SOURCE"], "fm")
            self.assertEqual(reset["environment"]["KNOWACT_KEEP_HISTORY"], "false")
            self.assertEqual(knowact["environment"]["KNOWACT_REPLAN_STEPS"], "16")
            self.assertTrue(baseline["matched_execution_interval"])

    def test_ar_baseline_can_be_selected_explicitly(self):
        rows = compare.build_plan(_args(g05_action_source="ar"))
        baseline = next(row for row in rows if row["model"] == "G05")
        self.assertEqual(baseline["environment"]["ROBODOJO_G05_ACTION_SOURCE"], "ar")

    def test_optional_memory_and_current_only_variants(self):
        args = _args(
            reset_ablation=True,
            g05_memory_ckpt="/checkpoints/g05_mem",
            g05_memory_offsets="-2,0",
            current_only_ckpt="/checkpoints/knowact_current",
            replan_steps=8,
            g05_execution_steps=8,
        )
        rows = {row["model"]: row for row in compare.build_plan(args)}
        self.assertEqual(set(rows), {"G05", "KnowAct", "KnowAct_reset", "G05_MEM", "KnowAct_current"})
        self.assertEqual(rows["G05_MEM"]["environment"]["ROBODOJO_G05_OBS_OFFSETS"], "-2,0")
        self.assertEqual(rows["G05_MEM"]["environment"]["ROBODOJO_G05_ACTION_SOURCE"], "fm")
        self.assertEqual(rows["KnowAct_current"]["environment"]["KNOWACT_KEEP_HISTORY"], "false")
        self.assertEqual(rows["KnowAct"]["execution_steps"], 8)
        self.assertTrue(rows["KnowAct"]["matched_execution_interval"])

    def test_memory_baseline_can_have_separate_action_source(self):
        args = _args(
            g05_memory_ckpt="/checkpoints/g05_mem",
            g05_memory_offsets="-2,0",
            g05_memory_action_source="ar",
        )
        rows = {row["model"]: row for row in compare.build_plan(args)}
        self.assertEqual(rows["G05"]["action_source"], "fm")
        self.assertEqual(rows["G05_MEM"]["action_source"], "ar")

    def test_unmatched_interval_requires_explicit_stress_flag(self):
        with self.assertRaisesRegex(ValueError, "execution intervals differ"):
            compare.build_plan(_args(replan_steps=8, g05_execution_steps=16))
        rows = compare.build_plan(
            _args(replan_steps=8, g05_execution_steps=16, allow_unmatched_interval=True)
        )
        knowact = next(row for row in rows if row["model"] == "KnowAct")
        self.assertFalse(knowact["matched_execution_interval"])
        self.assertEqual(knowact["environment"]["KNOWACT_ALLOW_STRIDE_SHIFT"], "true")

    def test_paired_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text(
                "task,seed,episode_id,model,success\n"
                "task_a,0,0,G05,0\n"
                "task_a,0,0,KnowAct,1\n"
                "task_a,1,0,G05,1\n"
                "task_a,1,0,KnowAct,1\n",
                encoding="utf-8",
            )
            result = compare.score_paired_episodes(path)
            self.assertEqual(result["KnowAct"]["paired_delta_vs_G05"], 0.5)
            self.assertEqual(result["KnowAct"]["task_seed_clusters"], 2)
            self.assertEqual(result["G05"]["success_rate"], 0.5)
            self.assertIsNotNone(result["KnowAct"]["cluster_bootstrap_95_interval"])

    def test_unpaired_episodes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text(
                "task,seed,episode_id,model,success\n"
                "task_a,0,0,G05,0\n"
                "task_a,1,0,KnowAct,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unpaired episodes"):
                compare.score_paired_episodes(path)

    def test_duplicate_episode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text(
                "task,seed,episode_id,model,success\n"
                "task_a,0,0,G05,0\n"
                "task_a,0,0,G05,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                compare.score_paired_episodes(path)


if __name__ == "__main__":
    unittest.main()
