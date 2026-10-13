"""Run the implemented CPU checks and retain the unfinished blueprint gates.

Run with the Python interpreter whose environment is being audited. This
script does not create an environment, migrate EVA, or load a checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from xpolicy.core.artifacts import file_sha256, source_revision, write_json

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    directories = ("src", "tests", "configs", "client_server", "utils", "policy/StarVLA", "tools")
    files = {ROOT / name for name in ("pyproject.toml", "uv.lock", "XPolicyLab.py", "model_template.py")}
    for directory in directories:
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"} and "__pycache__" not in path.parts)
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(files) if path.is_file()}


def run_check(identifier, command, output, timeout=120):
    started = time.monotonic()
    log = output / f"{identifier}.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        code, status = result.returncode, "PASS" if result.returncode == 0 else "FAIL"
    except (OSError, subprocess.TimeoutExpired) as exc:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{type(exc).__name__}: {exc}\n")
        code, status = None, "BLOCKED"
    return {"id": identifier, "status": status, "command": command, "cwd": str(ROOT), "exit_code": code, "duration_s": time.monotonic() - started, "log": str(log), "log_sha256": file_sha256(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", choices=("conda", "cpu"), required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    uv = shutil.which("uv")
    commands = [
        ("core_tests", [sys.executable, "-m", "pytest", "-q", "tests"]),
        ("cli_help", [sys.executable, "-m", "xpolicy", "--help"]),
        ("contract_doctor", [sys.executable, "-m", "xpolicy", "doctor", "--config", "configs/experiments/mock_contract.yaml", "--output", str(output / "doctor.json")]),
    ]
    if uv is not None:
        commands.append(("environment_dependencies", [uv, "pip", "check", "--python", sys.executable]))
    checks = [run_check(identifier, command, output) for identifier, command in commands]
    after = snapshot()
    checks.append({"id": "source_stability", "status": "PASS" if before == after else "FAIL", "message": "source hashes before and after checks must agree"})
    if uv is None:
        checks.append({"id": "environment_dependencies", "status": "BLOCKED", "message": "uv executable is missing"})
    inventory = sorted([{"name": distribution.metadata.get("Name", "unknown"), "version": distribution.version} for distribution in importlib.metadata.distributions()], key=lambda item: (item["name"].lower(), item["version"]))
    write_json(output / "software-inventory.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": inventory})
    checks.extend([
        {"id": "actual_training_loader_parity", "status": "NOT_RUN", "message": "core checks do not invoke a vendor loader; tools/export_g2_training_golden.py separately verifies genuine packed samples and adapter images, while final model tensors remain unverified"},
        {"id": "golden_dataset_direct_wire", "status": "NOT_RUN", "message": "HDF5 direct replay has synthetic checks and separate genuine raw-sample roundtrip evidence; the device wire path remains unverified"},
        {"id": "eva_robot_runtime", "status": "NOT_RUN", "message": "EVA migration deferred by user; no EVA source was changed or imported"},
        {"id": "original_benchmark_regression", "status": "BLOCKED", "message": "the outer benchmark client/env_cfg and a verified compatible checkpoint/dataset pairing are not configured"},
        {"id": "device_shadow_safety", "status": "BLOCKED", "message": "no device driver migration, verified SDK mapping or on-site acceptance evidence"},
        {"id": "external_asset_license_audit", "status": "NOT_RUN", "message": "core attribution is recorded; per-policy models, SDKs and robot assets still require separate audits"},
    ])
    implemented = [item for item in checks if "command" in item or item["id"] in {"environment_dependencies", "source_stability"}]
    statuses = {item["status"] for item in implemented}
    implementation_status = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    report = {
        "schema_version": 1, "scope": "unified_core", "environment": args.environment,
        "implementation_status": implementation_status, "blueprint_status": "INCOMPLETE",
        "provenance": {**source_revision(), "source_file_sha256": before, "source_file_sha256_after": after, "software_inventory_sha256": file_sha256(output / "software-inventory.json")},
        "checks": checks,
    }
    write_json(output / "TEST_REPORT.json", report)
    print(json.dumps({"report": str(output / "TEST_REPORT.json"), "implementation_status": implementation_status, "blueprint_status": report["blueprint_status"]}))
    return 0 if implementation_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
