import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_metadata", ROOT / "scripts/setup_local_eval_metadata.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


class EvaluationMetadataTests(unittest.TestCase):
    def test_registration_preserves_existing_environment_and_other_robots(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            registry = workspace / "env_cfg/robot/_robot_info.json"
            registry.parent.mkdir(parents=True)
            registry.write_text(json.dumps({"other": {"arm_dim": [3], "ee_dim": [1]}}))
            metadata.register_metadata(workspace, "arx_x5")
            env = workspace / "env_cfg/arx_x5.yml"
            original = yaml.safe_load(env.read_text())
            original["custom"] = "preserve me"
            env.write_text(yaml.safe_dump(original))
            metadata.register_metadata(workspace, "arx_x5")
            self.assertEqual(yaml.safe_load(env.read_text()), original)
            actual = json.loads(registry.read_text())
            self.assertIn("other", actual)
            training = json.loads((ROOT / "utils/robot/_robot_info.json").read_text())
            self.assertEqual(actual["arx_x5"], training["arx_x5"])

    def test_conflicting_robot_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "env_cfg/robot/_robot_info.json"
            registry.parent.mkdir(parents=True)
            original = '{"arx_x5": {"arm_dim": [3], "ee_dim": [1]}}'
            registry.write_text(original)
            with self.assertRaises(ValueError):
                metadata.register_metadata(temporary, "arx_x5")
            self.assertEqual(registry.read_text(), original)
            self.assertFalse((Path(temporary) / "env_cfg/arx_x5.yml").exists())

    def test_evaluator_prefix_with_spaces_and_explicit_interpreter(self):
        with tempfile.TemporaryDirectory(prefix="xpl eval ") as temporary:
            prefix = Path(temporary)
            (prefix / "bin").mkdir()
            (prefix / "bin/python").symlink_to(sys.executable)
            for requested in [str(prefix), sys.executable]:
                result = subprocess.run([
                    "bash", "-c", 'source "$1"; activate_eval_env "$2"; "$XPL_EVAL_PYTHON" -c "print(7)"',
                    "bash", str(ROOT / "utils/activate_eval_env.sh"), requested,
                ], check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout.strip(), "7")

    def test_client_setup_reads_quoted_yaml_path_with_requested_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            utilities = directory / "utils"
            utilities.mkdir()
            for name in ["activate_eval_env.sh", "resolve_eval_env_type.sh"]:
                shutil.copy2(ROOT / "utils" / name, utilities / name)
            (utilities / "run_debug_env_client.sh").write_text('printf "STUB_BATCH=%s\\n" "$1"\n')
            config = directory / 'policy "quoted".yml'
            config.write_text("eval_batch: false\nprotocol: ws\n")
            result = subprocess.run([
                "bash", str(ROOT / "utils/setup_env_client.sh"), str(utilities), str(config),
                sys.executable, "19999", "RoboDojo", "dummy", "arx_x5", "G05",
                "action_type=joint", str(directory), "0", "0", "localhost",
            ], env={**os.environ, "EVAL_ENV_TYPE": "debug"}, check=True,
                capture_output=True, text=True)
            self.assertIn("STUB_BATCH=false", result.stdout)
