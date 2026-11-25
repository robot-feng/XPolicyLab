import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv

import yaml


UTILS = Path(__file__).resolve().parents[1] / "utils"


class SimBenchmarkTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory(prefix="xpl workspace ")
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.env = self.root / "eval env"
        venv.EnvBuilder(symlinks=True).create(self.env)
        self.config = self.root / 'policy "example".yml'
        self.config.write_text("eval_batch: true\nprotocol: ws\n")
        self.named = self.root / "XPolicyLab/benchmarks/example/scripts/eval_policy.sh"
        self.legacy = self.root / "scripts/eval_policy.sh"
        self.legacy.parent.mkdir()
        self.legacy.write_text("echo legacy\n")

    def launch(self, name):
        return subprocess.run(
            ["bash", str(UTILS / "setup_env_client.sh"), str(UTILS), str(self.config),
             str(self.env), "6100", name, "pick red", "panda", "demo_policy",
             "ckpt_name=run 1,action_type=joint", str(self.root), "7", "2", "localhost"],
            env={**os.environ, "EVAL_ENV_TYPE": "sim", "PYTHONPATH": str(Path(yaml.__file__).resolve().parents[1])},
            capture_output=True, text=True,
        )

    def test_named_benchmark_receives_standard_arguments_in_selected_environment(self):
        external = self.root / "my simulator"
        (external / "scripts").mkdir(parents=True)
        install = self.named.parents[1]
        install.parent.mkdir(parents=True)
        install.symlink_to(external, target_is_directory=True)
        (external / "scripts/eval_policy.sh").write_text(
            "python -c 'import json, sys; print(json.dumps([sys.prefix, *sys.argv[1:]]))' \"$@\"\n"
        )
        run = self.launch("Example")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout.splitlines()[-1]), [
            str(self.env), "--bench_name", "Example", "--task_name", "pick red",
            "--env_cfg_type", "panda", "--policy_name", "demo_policy",
            "--host", "localhost", "--port", "6100", "--protocol", "ws",
            "--eval_batch", "true", "--root_dir", str(self.root), "--device_id", "2",
            "--additional_info", "ckpt_name=run 1,action_type=joint", "--seed", "7",
        ])

    def test_legacy_workspace_and_incomplete_installation(self):
        run = self.launch("RoboDojo")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines()[-1], "legacy")
        self.named.parent.mkdir(parents=True)
        run = self.launch("Example")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(str(self.named), run.stderr)
