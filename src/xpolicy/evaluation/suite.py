"""Sequential experiment suites with one ordinary Runner process per case/seed."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys
import threading
from typing import Annotated
from uuid import uuid4

from pydantic import Field, StrictBool, StringConstraints, model_validator
import yaml

from xpolicy.core.artifacts import file_sha256, source_revision, write_json
from xpolicy.core.config import ExperimentConfig, load_experiment, read_yaml, resolve_experiment_paths, resolve_path
from xpolicy.core.contracts import Name, Rate, Schema
from xpolicy.core.process import run_owned_process
from xpolicy.evaluation.lifecycle import StopSignal
from xpolicy.evaluation.reports import _read, summarize_reports, verify_report


CaseId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
Seed = Annotated[int, Field(ge=0, le=4294967295, strict=True)]


class SuiteCase(Schema):
    case_id: CaseId
    config: Path
    seeds: tuple[Seed, ...] = Field(min_length=1)
    repeats: int | None = Field(default=None, ge=1, strict=True)
    process_timeout_s: Rate | None = None
    python: Path | None = None

    @model_validator(mode="after")
    def unique_seeds(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("case seeds must be unique; use repeats for repeated trials")
        return self


class SuiteConfig(Schema):
    name: Name
    artifacts_root: Path
    process_timeout_s: Rate = 300.
    cleanup_grace_s: Annotated[float, Field(gt=0, le=30, strict=True, allow_inf_nan=False)] = 2.
    fail_fast: StrictBool = False
    cases: tuple[SuiteCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("suite case_id values must be unique")
        return self


def _digest(path):
    try:
        return file_sha256(path) if path.is_file() else None
    except OSError:
        return None


def _core_sources():
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package).as_posix(): _digest(path) for path in sorted(package.rglob("*.py"))}


def _runtime_lock():
    return _digest(Path(__file__).resolve().parents[3] / "uv.lock")


def _case_config(config, case, seed, directory):
    expected = config.model_dump(mode="json")
    expected["experiment"]["seed"] = seed
    if case.repeats is not None:
        expected["experiment"]["repeats"] = case.repeats
    expected["artifacts"]["root"] = str(directory / "artifacts")
    return ExperimentConfig.model_validate(expected).model_dump(mode="json")


def _contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("suite evidence must use relative contained paths")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("suite evidence escapes its run directory")
    return resolved


def _artifact_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.name == "SUITE_REPORT.json" and path.parent == root:
            continue
        if path.is_symlink() or not path.is_dir() and not path.is_file():
            raise ValueError("suite evidence contains a symlink or nonregular file")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_sha256(path)
    return result


def _plan(source, config, root):
    bindings, entries, case_paths = {str(source): file_sha256(source)}, [], {}
    (root / "source-suite.yaml").write_bytes(source.read_bytes())
    for case in config.cases:
        original = resolve_path(case.config, source.parent)
        case_paths[case.case_id] = str(original)
        bindings[str(original)] = _digest(original)
        python = case.python.expanduser() if case.python is not None else Path(sys.executable)
        # Preserve the executable symlink: resolving a venv's bin/python to
        # its base interpreter would silently select a different environment.
        python = (python if python.is_absolute() else source.parent / python).absolute()
        bindings[str(python)] = _digest(python)
        inputs = root / "inputs" / case.case_id
        inputs.mkdir(parents=True)
        prepared, error, profiles, resolutions = None, None, {}, {}
        try:
            (inputs / "source-experiment.yaml").write_bytes(original.read_bytes())
            prepared, contracts = load_experiment(original)
            snapshot = ExperimentConfig.model_validate(read_yaml(inputs / "source-experiment.yaml"))
            def record_path(path, parent):
                result = resolve_path(path, parent)
                resolutions[str(path)] = str(result)
                return result
            if resolve_experiment_paths(snapshot, original, resolver=record_path) != prepared:
                raise ValueError("experiment configuration/path resolution changed during suite planning")
            for name, path in (("manifest_sha256", contracts.manifest_path), ("robot_profile_sha256", contracts.robot_path),
                               ("dataset_profile_sha256", contracts.dataset_profile_path)):
                profiles[name] = file_sha256(path) if path is not None else None
                if path is not None:
                    bindings[str(path)] = profiles[name]
                    (inputs / f"{name}.yaml").write_bytes(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        for seed in case.seeds:
            relative = Path("cases") / case.case_id / f"seed-{seed}"
            directory = root / relative
            directory.mkdir(parents=True)
            generated, expected, repeats = None, None, None
            if error is None:
                expected = _case_config(prepared, case, seed, directory)
                repeats = expected["experiment"]["repeats"]
                generated = (relative / "experiment.yaml").as_posix()
                (root / generated).write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")
            entries.append({"case_id": case.case_id, "seed": seed, "requested_repeats": repeats,
                            "python": str(python), "interpreter_sha256": _digest(python),
                            "source_config": str(original), "directory": relative.as_posix(), "config": generated,
                            "configuration_sha256": _digest(root / generated) if generated else None,
                            "profile_sha256": profiles, "expected_config": expected, "preparation_error": error,
                            "path_resolutions": resolutions,
                            "process_timeout_s": case.process_timeout_s or config.process_timeout_s})
    return {"schema_version": 1, "name": config.name, "source_path": str(source), "artifacts_root": str(root),
            "case_config_resolutions": case_paths, "cases": entries, "source_sha256": bindings,
            "core_source_sha256": _core_sources(), "runtime_lock_sha256": _runtime_lock(),
            "suite_settings": config.model_dump(mode="json")}


def _result(root, entry, core_sources):
    directory = _contained(root, entry["directory"])
    paths = list((directory / "artifacts").glob("*/results.json"))
    if not paths:
        return None, None
    if len(paths) != 1:
        raise ValueError("one case process must produce exactly one experiment result")
    path = _contained(root, paths[0].relative_to(root).as_posix())
    result = verify_report(path)
    actual = _read(path.parent / "config.json")
    if actual != entry["expected_config"]:
        raise ValueError("child result configuration differs from the planned case/seed")
    expected_hashes = {"configuration_sha256": entry["configuration_sha256"], **entry["profile_sha256"]}
    if any(result["provenance"].get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("child configuration/profile identities differ from the suite plan")
    if result["provenance"]["core_source_sha256"] != core_sources:
        raise ValueError("child Runner core source differs from the suite plan")
    return path, result


def _case_status(process, result):
    if not process["root_reaped"]:
        return "BLOCKED"
    if process["timed_out"]:
        return "TIMED_OUT"
    if process["stop_reason"] in {"operator_stop", "interrupted"}:
        return "STOPPED"
    if result is None or process["exit_code"] != (0 if result["status"] == "COMPLETED" else 2):
        return "FAIL"
    return result["status"]


def _summary(paths, root):
    if not paths:
        return None
    summary = summarize_reports(paths)
    for item in summary["inputs"]:
        item["path"] = Path(item["path"]).relative_to(root).as_posix()
    return summary


def _status(report):
    states = {case["status"] for case in report["cases"]}
    if report["errors"] or not report["source_stable"] or states & {"FAIL", "FAULT", "TIMED_OUT"} or report["summary"] and report["summary"]["status"] == "FAIL":
        return "FAIL"
    if "BLOCKED" in states:
        return "BLOCKED"
    if report["stop_request"] or states & {"STOPPED", "NOT_RUN"}:
        return "STOPPED"
    return "PASS"


def _stable(plan, report):
    return (bool(plan["core_source_sha256"]) and all(digest is not None for digest in plan["core_source_sha256"].values()) and
            plan["source_sha256"] == report["source_sha256_after"] and
            plan["core_source_sha256"] == report["core_source_sha256_after"] and
            plan["runtime_lock_sha256"] == report["runtime_lock_sha256_after"])


def _evaluate(source, stop_signal):
    source = source.expanduser().resolve()
    config = SuiteConfig.model_validate(read_yaml(source))
    root = resolve_path(config.artifacts_root, source.parent)
    root.mkdir(parents=True, exist_ok=False)
    plan = _plan(source, config, root)
    write_json(root / "plan.json", plan)
    report = {"schema_version": 1, "scope": "isolated_experiment_suite", "suite_id": uuid4().hex,
              "name": config.name, "status": "RUNNING", "provenance": source_revision(),
              "plan_sha256": file_sha256(root / "plan.json"), "cases": [], "errors": [],
              "not_verified": ["causal_task_success", "simulator_regression", "EVA_or_hardware", "physical_stop"]}
    for entry in plan["cases"]:
        report["cases"].append({key: entry[key] for key in ("case_id", "seed", "requested_repeats", "directory", "config")})
        report["cases"][-1].update(status="NOT_RUN", reason="not_scheduled", process_pid=None, process=None, result=None, result_sha256=None)
    write_json(root / "SUITE_REPORT.json", report)
    verified, halt_reason = [], None
    try:
        for entry, case in zip(plan["cases"], report["cases"]):
            if stop_signal.request_info is not None:
                halt_reason = stop_signal.request_info["reason"]
            if halt_reason is not None:
                case["reason"] = halt_reason
                continue
            if any(_digest(Path(path)) != digest for path, digest in plan["source_sha256"].items()):
                raise ValueError("suite source configuration/profile bytes changed before scheduling")
            if entry["preparation_error"] is not None:
                case.update(status="BLOCKED", reason="case_configuration_error", error=entry["preparation_error"])
            else:
                case.update(status="RUNNING", reason=None)
                command = [entry["python"], "-B", "-m", "xpolicy", "eval", "--config", str(root / entry["config"])]
                case["command"] = command
                def started(pid):
                    case["process_pid"] = pid
                    case["active_pid"] = pid
                    write_json(root / "SUITE_REPORT.json", report)
                try:
                    process = run_owned_process(command, Path.cwd(), dict(os.environ), entry["process_timeout_s"],
                                                root / entry["directory"] / "process.log", stop_signal=stop_signal,
                                                interrupt_first=True, grace_s=config.cleanup_grace_s, on_started=started)
                    case["process"] = process
                    if not process["root_reaped"]:
                        halt_reason = "process_cleanup_pending"
                    path, result = _result(root, entry, plan["core_source_sha256"])
                    if path is not None:
                        case.update(result=path.relative_to(root).as_posix(), result_sha256=file_sha256(path), run_outcome=result["status"])
                        verified.append(path)
                    else:
                        case["error"] = {"type": "MissingExperimentResult", "message": "child did not produce a verified experiment result; inspect process.log"}
                    case["status"] = _case_status(process, result)
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    case.update(status="BLOCKED" if case["process_pid"] is None else "FAIL",
                                reason="process_start_failed" if case["process_pid"] is None else "case_evidence_failure",
                                error={"type": type(exc).__name__, "message": str(exc)})
                finally:
                    case.pop("active_pid", None)
            if config.fail_fast and case["status"] != "COMPLETED":
                halt_reason = halt_reason or "prior_case_failure"
            write_json(root / "SUITE_REPORT.json", report)
    except KeyboardInterrupt:
        stop_signal.request("interrupted")
        for case in report["cases"]:
            if case["status"] == "RUNNING":
                case.update(status="STOPPED", reason="interrupted")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        for case in report["cases"]:
            if case["status"] == "RUNNING":
                case.update(status="FAIL", reason="supervisor_error")
            if case["status"] == "NOT_RUN" and case["reason"] == "not_scheduled":
                case["reason"] = "suite_stopped_before_schedule"
        report["stop_request"] = stop_signal.request_info
        report["source_sha256_after"] = {path: _digest(Path(path)) for path in plan["source_sha256"]}
        report["core_source_sha256_after"] = _core_sources()
        report["runtime_lock_sha256_after"] = _runtime_lock()
        report["source_stable"] = _stable(plan, report)
        report["summary"] = _summary(verified, root)
        report["population"] = {"planned_cases": len(report["cases"]),
                                "processes_started": sum(case["process_pid"] is not None for case in report["cases"]),
                                "outcome_counts": dict(Counter(case["status"] for case in report["cases"])),
                                "verified_runs": report["summary"]["population"]["valid_runs"] if report["summary"] else 0}
        try:
            report["evidence_sha256"] = _artifact_hashes(root)
        except (OSError, ValueError) as exc:
            report["evidence_sha256"] = {}
            report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        report["status"] = _status(report)
        write_json(root / "SUITE_REPORT.json", report)
    return report


def _verify_suite(path):
    """Verify a closed suite, its plan, nested run evidence and recomputed summary."""
    path = Path(path).expanduser().resolve()
    root, report = path.parent, _read(path)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1 or report.get("scope") != "isolated_experiment_suite":
        raise ValueError("expected a version 1 isolated experiment suite")
    if report["status"] not in {"PASS", "FAIL", "BLOCKED", "STOPPED"}:
        raise ValueError("suite has not reached a terminal outcome")
    if report["evidence_sha256"] != _artifact_hashes(root):
        raise ValueError("suite evidence SHA256 mismatch")
    if not {"plan.json", "source-suite.yaml"} <= report["evidence_sha256"].keys() or report["plan_sha256"] != file_sha256(root / "plan.json"):
        raise ValueError("suite plan identity is missing or changed")
    plan = _read(root / "plan.json")
    settings = SuiteConfig.model_validate(plan["suite_settings"])
    if SuiteConfig.model_validate(read_yaml(root / "source-suite.yaml")) != settings or file_sha256(root / "source-suite.yaml") != plan["source_sha256"][plan["source_path"]]:
        raise ValueError("suite settings differ from the retained source configuration")
    planned_ids = [(case.case_id, seed) for case in settings.cases for seed in case.seeds]
    if report["name"] != settings.name or plan["name"] != settings.name or [(entry["case_id"], entry["seed"]) for entry in plan["cases"]] != planned_ids:
        raise ValueError("suite plan differs from its declared case/seed population")
    if len(report["cases"]) != len(plan["cases"]):
        raise ValueError("suite case denominator differs from its plan")
    stable = _stable(plan, report)
    if type(report["source_stable"]) is not bool or report["source_stable"] != stable:
        raise ValueError("suite source stability disagrees with retained identities")
    paths = []
    declared = {case.case_id: case for case in settings.cases}
    historical_root = Path(plan["artifacts_root"])
    if not historical_root.is_absolute():
        raise ValueError("historical suite artifact root must be absolute")
    for entry, case in zip(plan["cases"], report["cases"]):
        declaration = declared[entry["case_id"]]
        expected_directory = f"cases/{entry['case_id']}/seed-{entry['seed']}"
        expected_source = Path(plan["case_config_resolutions"][entry["case_id"]])
        if not expected_source.is_absolute():
            raise ValueError("historical source config resolution must be absolute")
        if entry["directory"] != expected_directory or entry["source_config"] != str(expected_source):
            raise ValueError("planned source/configuration directory differs from its suite declaration")
        if any(case[key] != entry[key] for key in ("case_id", "seed", "requested_repeats", "directory", "config")):
            raise ValueError("suite case identity/configuration differs from its plan")
        if entry["config"] is not None and file_sha256(_contained(root, entry["config"])) != entry["configuration_sha256"]:
            raise ValueError("planned case configuration bytes changed")
        if entry["config"] is not None and ExperimentConfig.model_validate(read_yaml(_contained(root, entry["config"]))).model_dump(mode="json") != entry["expected_config"]:
            raise ValueError("planned effective configuration differs from its YAML snapshot")
        source_digest = plan["source_sha256"][entry["source_config"]]
        if source_digest is not None and file_sha256(_contained(root, f"inputs/{entry['case_id']}/source-experiment.yaml")) != source_digest:
            raise ValueError("source experiment snapshot differs from its planned identity")
        for name, digest in entry["profile_sha256"].items():
            if digest is not None and file_sha256(_contained(root, f"inputs/{entry['case_id']}/{name}.yaml")) != digest:
                raise ValueError("profile snapshot differs from its planned identity")
        if entry["config"] is not None:
            if entry["config"] != f"{expected_directory}/experiment.yaml":
                raise ValueError("generated configuration path differs from its declared case/seed")
            snapshot = ExperimentConfig.model_validate(read_yaml(_contained(root, f"inputs/{entry['case_id']}/source-experiment.yaml")))
            def historical_path(path, parent):
                # Resolve from captured receipts, never from another host's
                # current symlinks, home directory, datasets or checkpoints.
                resolved = Path(entry["path_resolutions"][str(path)])
                if not resolved.is_absolute():
                    raise ValueError("historical path resolution must be absolute")
                return resolved
            resolved = resolve_experiment_paths(snapshot, expected_source, resolver=historical_path)
            expected = _case_config(resolved, declaration, entry["seed"], historical_root / expected_directory)
            if entry["expected_config"] != expected or entry["requested_repeats"] != expected["experiment"]["repeats"]:
                raise ValueError("effective case overrides differ from the declared source experiment")
            for name, original in (("manifest_sha256", resolved.policy.manifest), ("robot_profile_sha256", resolved.environment.robot),
                                   ("dataset_profile_sha256", resolved.environment.dataset_profile)):
                if entry["profile_sha256"].get(name) != (plan["source_sha256"].get(str(original)) if original is not None else None):
                    raise ValueError("profile identity differs from the resolved source experiment")
        if case["status"] not in {"NOT_RUN", "COMPLETED", "FAIL", "FAULT", "BLOCKED", "TIMED_OUT", "STOPPED"}:
            raise ValueError("unknown suite case outcome")
        if case["process"] is None:
            if case["status"] == "COMPLETED" or case["result"] is not None or case["result_sha256"] is not None:
                raise ValueError("unlaunched case cannot claim a completed or verified run")
            if case["process_pid"] is None:
                if case["status"] == "BLOCKED" and entry["preparation_error"] is not None and case.get("error") == entry["preparation_error"]:
                    continue
                if case["status"] == "BLOCKED" and case["reason"] == "process_start_failed" and case.get("error"):
                    continue
                if case["status"] != "NOT_RUN":
                    raise ValueError("unlaunched case outcome lacks a preparation failure")
            continue
        process = case["process"]
        if process["pid"] != case["process_pid"] or type(process["root_reaped"]) is not bool or type(process["timed_out"]) is not bool:
            raise ValueError("invalid process execution receipt")
        if process["timeout_s"] != entry["process_timeout_s"] or process["cleanup_grace_s"] != settings.cleanup_grace_s:
            raise ValueError("process deadline differs from its suite plan")
        if type(process["pid"]) is not int or process["pid"] < 1 or process["root_reaped"] != (process["exit_code"] is not None):
            raise ValueError("process identity or reaping outcome is invalid")
        if process["stop_reason"] not in {None, "process_timeout", "operator_stop", "interrupted"} or process["timed_out"] != (process["stop_reason"] == "process_timeout"):
            raise ValueError("process stop and timeout receipts disagree")
        result_path, result = _result(root, entry, plan["core_source_sha256"])
        if case["status"] != _case_status(process, result):
            raise ValueError("suite case outcome differs from its process/run evidence")
        relative = result_path.relative_to(root).as_posix() if result_path is not None else None
        if case["result"] != relative or case["result_sha256"] != (file_sha256(result_path) if result_path else None):
            raise ValueError("suite child result binding is missing or changed")
        if result_path is not None:
            if case["run_outcome"] != result["status"]:
                raise ValueError("suite child outcome differs from the verified run")
            paths.append(result_path)
    summary = _summary(paths, root)
    population = {"planned_cases": len(report["cases"]), "processes_started": sum(case["process_pid"] is not None for case in report["cases"]),
                  "outcome_counts": dict(Counter(case["status"] for case in report["cases"])),
                  "verified_runs": summary["population"]["valid_runs"] if summary else 0}
    if report["summary"] != summary or report["population"] != population or report["status"] != _status(report):
        raise ValueError("suite summary, population or outcome differs from retained evidence")
    return report


def verify_suite(path):
    """Reject malformed or changed suites before trusting their population."""
    try:
        return _verify_suite(path)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError(f"invalid suite evidence: {exc}") from exc


class ExperimentSuiteRunner:
    """Evaluate once; each case keeps the existing CLI/Runner admission path."""

    def __init__(self, path):
        self.path, self.stop_signal = Path(path), StopSignal()
        self._lock = threading.Lock()
        self._used, self._report, self._error = False, None, None

    def stop(self, reason="operator_stop"):
        return self.stop_signal.request(reason)

    def evaluate(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this ExperimentSuiteRunner already owns an active evaluation")
        try:
            if self._used:
                if self._error is not None:
                    raise self._error
                return self._report
            self._used = True
            try:
                self._report = _evaluate(self.path, self.stop_signal)
                return self._report
            except BaseException as exc:
                self._error = exc
                raise
        finally:
            self._lock.release()
