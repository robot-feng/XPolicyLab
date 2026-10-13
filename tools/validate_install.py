"""Build a wheel and audit it from a fresh environment, with evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from validate_core import ROOT, run_check, snapshot
from xpolicy.core.artifacts import file_sha256, source_revision, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    before, checks = snapshot(), []
    if uv is None:
        checks.append({"id": "uv_available", "status": "BLOCKED", "message": "uv executable is missing"})
    else:
        checks.append(run_check("build_wheel", [uv, "build", "--wheel", "--out-dir", str(output / "dist"), "--python", sys.executable], output))
        checks.append(run_check("export_core_lock", [uv, "export", "--locked", "--no-emit-project", "--default-index", "https://pypi.org/simple", "--python", sys.executable, "--output-file", str(output / "requirements.txt")], output))
        if all(check["status"] == "PASS" for check in checks):
            wheel = output / "dist/xpolicylab-0.0.1-py3-none-any.whl"
            with tempfile.TemporaryDirectory(prefix="xpolicy-wheel-") as temporary:
                environment = Path(temporary) / "env"
                python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                steps = [
                    ("create_clean_environment", [uv, "venv", str(environment), "--python", sys.executable]),
                    ("install_locked_dependencies", [uv, "pip", "install", "--python", str(python), "--require-hashes", "--requirements", str(output / "requirements.txt")]),
                    ("install_wheel", [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
                    ("isolated_import", [str(python), "-I", "-c", "import importlib.util, json, sys, xpolicy; from importlib.metadata import distributions; from xpolicy.core.artifacts import source_revision; from xpolicy.core.registry import policy_registry; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('rospy') is None; assert not any(n.split('.')[0] in {'torch','transformers','rospy','gdk'} for n in sys.modules); assert 'site-packages' in xpolicy.__file__; assert [e.name for e in policy_registry().entries()] == ['mock']; assert source_revision()['git_sha'] is None; print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': sorted([{'name': d.metadata.get('Name', 'unknown'), 'version': d.version} for d in distributions()], key=lambda d: d['name'])}))"]),
                    ("wheel_cli_help", [str(python), "-I", "-m", "xpolicy", "--help"]),
                    ("wheel_doctor", [str(python), "-I", "-m", "xpolicy", "doctor", "--config", str(ROOT / "configs/experiments/mock_contract.yaml"), "--output", str(output / "doctor.json")]),
                    ("wheel_dependencies", [uv, "pip", "check", "--python", str(python)]),
                ]
                for identifier, command in steps:
                    result = run_check(identifier, command, output)
                    checks.append(result)
                    if result["status"] != "PASS":
                        break
            checks.append({"id": "wheel_identity", "status": "PASS", "sha256": file_sha256(wheel), "path": str(wheel)})
    after = snapshot()
    checks.append({"id": "source_stability", "status": "PASS" if before == after else "FAIL"})
    statuses = {check["status"] for check in checks}
    status = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    report = {"schema_version": 1, "scope": "A01_clean_public_wheel", "status": status, "provenance": {**source_revision(), "source_file_sha256": before, "source_file_sha256_after": after}, "checks": checks}
    write_json(output / "TEST_REPORT.json", report)
    print(json.dumps({"report": str(output / "TEST_REPORT.json"), "status": status}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
