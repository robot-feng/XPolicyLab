"""Record what contract preflight proves and what has not been run."""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement

from xpolicy.core.artifacts import file_sha256, source_revision, verify_artifact
from xpolicy.core.config import load_experiment
from xpolicy.core.contracts import ActionType, ContractValidator
from xpolicy.core.registry import policy_registry


def dependency_check() -> dict:
    versions, errors = {}, []
    for text in importlib.metadata.requires("xpolicylab") or []:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            version = importlib.metadata.version(requirement.name)
            versions[requirement.name] = version
            if version not in requirement.specifier:
                errors.append(f"{requirement.name}=={version} does not satisfy {requirement.specifier}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing dependency: {requirement.name}")
    return {"id": "public_dependencies", "status": "FAIL" if errors else "PASS", "details": {"versions": versions, "errors": errors}}


def doctor(path: str | Path, *, verify_artifacts: bool = False) -> dict:
    source = Path(path).expanduser().resolve()
    checks = []
    provenance = source_revision()
    try:
        checks.append(dependency_check())
    except importlib.metadata.PackageNotFoundError:
        checks.append({"id": "public_dependencies", "status": "FAIL", "message": "install the project with uv pip install --python <conda-python> -e . first"})
    try:
        config, contracts = load_experiment(source)
        checks.append({"id": "configuration_schema", "status": "PASS"})
        provenance["configuration_sha256"] = file_sha256(source)
        provenance["profile_sha256"] = {
            str(item): file_sha256(item)
            for item in (contracts.manifest_path, contracts.robot_path, contracts.dataset_profile_path)
            if item is not None
        }
        issues = ContractValidator.profiles(contracts.manifest, contracts.robot, contracts.dataset)
        checks.append({"id": "training_dataset_robot_contract", "status": "FAIL" if issues else "PASS", "details": {"issues": [asdict(item) for item in issues]}})
        try:
            entry = policy_registry().get(config.policy.name)
            supported = config.policy.api in entry.capabilities
            checks.append({"id": "policy_api", "status": "PASS" if supported else "BLOCKED", "details": {"name": entry.name, "requested": config.policy.api, "capabilities": entry.capabilities}})
            if entry.name == "mock" and contracts.manifest.source_kind != "mock":
                checks.append({"id": "policy_manifest_origin", "status": "FAIL", "message": "mock policy requires a mock manifest"})
            elif entry.name != "mock" and contracts.manifest.source_kind != "checkpoint":
                checks.append({"id": "policy_manifest_origin", "status": "FAIL", "message": "model adapter requires a checkpoint manifest"})
        except ValueError as exc:
            checks.append({"id": "policy_api", "status": "BLOCKED", "message": str(exc)})
        horizon_ok = config.execution.execute_horizon <= contracts.manifest.predicted_horizon
        rate_ok = config.execution.control_hz == contracts.robot.control_hz
        blend_ok = not config.execution.overlap_blend_steps or contracts.manifest.action_type == ActionType.JOINT_POSITION
        dataset_rate_ok = contracts.dataset is None or config.execution.control_hz == contracts.dataset.sample_hz
        checks.append({"id": "execution_schedule", "status": "PASS" if horizon_ok and rate_ok and blend_ok and dataset_rate_ok else "FAIL", "details": {
            "horizon_within_prediction": horizon_ok, "control_rate_matches_robot": rate_ok,
            "control_rate_matches_dataset": dataset_rate_ok, "overlap_requires_absolute_joint_position": blend_ok,
            "strategy": config.execution.strategy, "latency_trim": config.execution.latency_trim}})
        identities = [(name, getattr(contracts.manifest, name), contracts.manifest_path) for name in ("checkpoint", "training_loader", "normalization_stats") if getattr(contracts.manifest, name) is not None]
        if contracts.dataset is not None:
            identities.append(("dataset", contracts.dataset.artifact, contracts.dataset_profile_path))
        for name, identity, declaring_file in identities:
            if verify_artifacts:
                try:
                    actual_path = verify_artifact(identity, declaring_file)
                    checks.append({"id": f"artifact_hash.{name}", "status": "PASS", "details": {"path": str(actual_path), "sha256": identity.sha256}})
                except ValueError as exc:
                    checks.append({"id": f"artifact_hash.{name}", "status": "FAIL", "message": str(exc)})
                except (OSError, RuntimeError) as exc:
                    checks.append({"id": f"artifact_hash.{name}", "status": "BLOCKED", "message": str(exc)})
            else:
                checks.append({"id": f"artifact_hash.{name}", "status": "NOT_RUN", "message": "identity declared; use --verify-artifacts to verify local bytes"})
        checks.append({"id": "environment_runtime", "status": "NOT_RUN", "message": f"contract mode does not run {config.environment.kind} or measure task success"})
        if config.environment.kind in {"shadow", "real"}:
            checks.append({"id": "hardware_admission", "status": "BLOCKED", "message": "no migrated device driver, calibration or on-site acceptance evidence is available"})
    except (OSError, ValueError, TypeError) as exc:
        checks.append({"id": "configuration_schema", "status": "FAIL", "message": str(exc)})
    statuses = {item["status"] for item in checks}
    status = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    return {
        "schema_version": 1, "scope": "contract_preflight", "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": str(source), "provenance": provenance, "checks": checks,
    }
