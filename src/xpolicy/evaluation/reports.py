"""Verify versioned run evidence and aggregate comparable trial populations."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from xpolicy.core.artifacts import file_sha256
from xpolicy.core.config import ExperimentConfig
from xpolicy.core.contracts import ModelManifest, RobotProfile
from xpolicy.core.tasks import resolve_task
from xpolicy.evaluation.metrics import series
from xpolicy.evaluation.lifecycle import NEXT, TrialState


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique, parse_constant=_constant)


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("report identities must be nonempty strings")
    return value


def _validate(path):
    report = _read(path)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 2:
        raise ValueError("summary requires schema_version=2 with evidence hashes; legacy reports must be rerun")
    json.dumps(report, allow_nan=False)
    _name(report["run_id"])
    if report["status"] not in {"COMPLETED", "FAULT", "BLOCKED", "STOPPED"}:
        raise ValueError("unknown run outcome")
    for key in ("environment_kind", "execution_mode", "execution_strategy", "task", "conditions", "provenance"):
        if key not in report:
            raise ValueError(f"report is missing {key}")
    task_id = _name(report["task"]["task_id"])
    evidence = report["evidence_sha256"]
    if not isinstance(evidence, dict) or not {"trace.jsonl", "config.json", "profiles.json", "software-inventory.json"} <= evidence.keys():
        raise ValueError("report lacks mandatory evidence hashes")
    for name, digest in evidence.items():
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise ValueError("evidence paths must be filenames beside results.json")
        item = (path.parent / name).resolve()
        if item.parent != path.parent.resolve() or file_sha256(item) != digest:
            raise ValueError(f"evidence SHA256 mismatch: {name}")
    config = ExperimentConfig.model_validate(_read(path.parent / "config.json"))
    profiles = _read(path.parent / "profiles.json")
    manifest, robot = ModelManifest.model_validate(profiles["model"]), RobotProfile.model_validate(profiles["robot"])
    expected_conditions = {"policy_id": manifest.model_id, "policy_options": config.policy.options,
                           "robot_profile_id": robot.profile_id, "execution": config.execution.model_dump(mode="json"),
                           "effective_timeout_s": min(config.experiment.max_duration_s, config.task.timeout_s if config.task and config.task.timeout_s else config.experiment.max_duration_s)}
    if type(report["provenance"]["seed"]) is not int:
        raise ValueError("report seed must be an integer")
    if report["conditions"] != expected_conditions or report["task"] != resolve_task(config, manifest).model_dump(mode="json"):
        raise ValueError("report conditions or task differ from hashed config/profile snapshots")
    if (report["environment_kind"], report["execution_mode"], report["execution_strategy"], report["provenance"]["seed"], report["requested_repeats"]) != (config.environment.kind, config.execution.mode, config.execution.strategy, config.experiment.seed, config.experiment.repeats):
        raise ValueError("report population or mode differs from hashed configuration")
    ends, transitions, providers = {}, {}, []
    for line in (path.parent / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line, object_pairs_hook=_unique, parse_constant=_constant)
        if event["run_id"] != report["run_id"]:
            raise ValueError("trace run identity differs from results")
        if event["stage"] == "trial_state":
            transitions.setdefault(event["trial_id"], []).append({key: event[key] for key in ("from_state", "state", "monotonic_s")})
        if event["stage"] == "environment_prepared":
            providers.append(event["provider"])
        if event["stage"] == "trial_end":
            identity = event["trial_id"]
            if identity in ends:
                raise ValueError("duplicate trial_end evidence")
            ends[identity] = event["result"]
    trials, not_started = report["trials"], report["not_started_trials"]
    if "environment_provider" in report:
        expected_provider = report["environment_provider"]
        if expected_provider is None and providers or expected_provider is not None and (not providers or any(item != expected_provider for item in providers)):
            raise ValueError("environment provider differs from hashed preparation evidence")
    if not isinstance(trials, list) or not isinstance(not_started, list):
        raise ValueError("trial populations must be lists")
    if type(report["requested_repeats"]) is not int or report["requested_repeats"] < 1 or len(trials) + len(not_started) != report["requested_repeats"]:
        raise ValueError("requested repeat denominator differs from retained trial populations")
    identities = []
    unstarted_ids = {trial["trial_id"] for trial in not_started}
    for trial in trials + not_started:
        identity = _name(trial["trial_id"])
        identities.append(identity)
        if trial["task_id"] != task_id:
            raise ValueError("trial task identity differs from run")
        expected = dict(trial)
        if trial["trial_id"] in unstarted_ids:
            if trial["status"] != "NOT_RUN" or trial["lifecycle"]["started"] is not False:
                raise ValueError("unstarted trial must be NOT_RUN")
            expected.pop("reason")
            expected["status"] = "BLOCKED"
        if ends.get(identity) != expected:
            raise ValueError(f"trial result differs from hashed trial_end trace: {identity}")
        history = trial["lifecycle"]["history"]
        if transitions.get(identity) != history or not history or history[0]["state"] != "CREATED" or history[0]["from_state"] is not None:
            raise ValueError("lifecycle history differs from actual state transition trace")
        for previous, current in zip(history, history[1:]):
            if current["from_state"] != previous["state"] or TrialState(current["state"]) not in NEXT.get(TrialState(previous["state"]), set()) or current["monotonic_s"] < previous["monotonic_s"]:
                raise ValueError("trial lifecycle contains an illegal or nonmonotonic transition")
        if trial["lifecycle"]["state"] != history[-1]["state"] or trial["lifecycle"]["started"] != any(item["state"] == "RUNNING" for item in history):
            raise ValueError("trial readiness/final state disagrees with lifecycle history")
        if type(trial["causal"]) is not bool or type(trial["lifecycle"]["started"]) is not bool:
            raise ValueError("causality and lifecycle readiness require booleans")
        success, source = trial["task_success"], trial["success_source"]
        if source not in {"not_measured", "human", "ground_truth", "heuristic"}:
            raise ValueError("unknown task judge source")
        if (success is None) != (source == "not_measured") or (success is not None and type(success) is not bool):
            raise ValueError("judged success requires a boolean and explicit judge source")
        if not trial["causal"] and success is not None:
            raise ValueError("noncausal replay cannot claim current-policy task success")
        if trial["status"] not in {"COMPLETED", "FAULT", "STOPPED", "BLOCKED", "NOT_RUN"}:
            raise ValueError("unknown trial outcome")
        series([trial["duration_s"]])
        for metric in trial["metrics"].values():
            _name(metric["unit"]); _name(metric["scope"])
            values = metric["values"]
            if any(type(value) not in (float, int) for value in values) or type(metric["samples"]) is not int:
                raise ValueError("metric samples require real numbers and an integer count")
            if metric != series(values, unit=metric["unit"], scope=metric["scope"]):
                raise ValueError("metric statistics differ from their raw samples")
    if len(set(identities)) != len(identities) or set(ends) != set(identities):
        raise ValueError("trial identities must be unique and exhaust the trace population")
    statuses = {trial["status"] for trial in trials}
    provenance = report["provenance"]
    stable = (provenance["core_source_sha256"] == provenance["core_source_sha256_after"] and
              provenance["runtime_lock_sha256"] == provenance["runtime_lock_sha256_after"] and
              all(provenance[key] == digest for key, digest in provenance["input_sha256_after"].items()))
    if type(report["source_stable"]) is not bool or report["source_stable"] != stable or bool(report["evidence_errors"]) == stable:
        raise ValueError("source stability outcome disagrees with retained before/after identities")
    expected_status = "FAULT" if report["cleanup_errors"] or report["evidence_errors"] or "FAULT" in statuses else "BLOCKED" if "BLOCKED" in statuses else "STOPPED" if "STOPPED" in statuses or not_started else "COMPLETED"
    if report["status"] != expected_status:
        raise ValueError("run status disagrees with retained trial/cleanup outcomes")
    return report


def summarize_reports(paths):
    """Keep failed inputs visible; pool raw latency samples only within conditions."""
    inputs, populations, seen_runs, seen_trials = [], [], set(), set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        try:
            report = _validate(path)
            if report["run_id"] in seen_runs:
                raise ValueError("duplicate run_id would count the same experiment twice")
            identities = {trial["trial_id"] for trial in report["trials"] + report["not_started_trials"]}
            if seen_trials & identities:
                raise ValueError("duplicate trial identities across reports")
            seen_runs.add(report["run_id"]); seen_trials.update(identities)
            inputs.append({"path": str(path), "sha256": file_sha256(path), "status": report["status"], "run_id": report["run_id"], "integrity": "VERIFIED"})
            populations.append(report)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            inputs.append({"path": str(path), "status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}})
    if not inputs:
        raise ValueError("at least one report is required")
    groups = {}
    for report in populations:
        provenance = report["provenance"]
        common = {"environment_kind": report["environment_kind"], "execution_mode": report["execution_mode"],
                  "execution_strategy": report["execution_strategy"], "task": report["task"], "conditions": report["conditions"],
                  "software_inventory_sha256": report["evidence_sha256"]["software-inventory.json"],
                  "identities": {key: provenance.get(key) for key in ("manifest_sha256", "robot_profile_sha256", "dataset_profile_sha256", "dataset_sha256", "checkpoint_sha256", "core_source_sha256", "runtime_lock_sha256")}}
        for trial in report["trials"]:
            condition = {**common, "success_source": trial["success_source"]}
            key = hashlib.sha256(json.dumps(condition, sort_keys=True, allow_nan=False).encode()).hexdigest()
            group = groups.setdefault(key, {"condition_id": key, "conditions": condition, "runs": set(), "seeds": set(), "trials": []})
            group["runs"].add(report["run_id"]); group["seeds"].add(provenance["seed"])
            group["trials"].append(trial)
    result_groups = []
    for group in groups.values():
        trials = group.pop("trials")
        outcomes = Counter(item["status"] for item in trials)
        started = [item for item in trials if item["lifecycle"]["started"]]
        judged = [item for item in trials if item["task_success"] is not None]
        successes = sum(item["task_success"] for item in judged)
        metrics = {}
        for trial in trials:
            for name, metric in trial["metrics"].items():
                signature = (name, metric["unit"], metric["scope"])
                metrics.setdefault(signature, []).extend(metric["values"])
        result_groups.append({**group, "runs": sorted(group["runs"]), "seeds": sorted(group["seeds"]),
            "trials": len(trials), "execution_started_trials": len(started), "outcome_counts": dict(outcomes),
            "execution_failure_rate": sum(item["status"] == "FAULT" for item in started) / len(started) if started else None,
            "admission_failures": sum(item["status"] in {"FAULT", "BLOCKED"} and not item["lifecycle"]["started"] for item in trials),
            "task_score": {"judged_trials": len(judged), "unjudged_trials": len(trials) - len(judged),
                           "successes": successes, "success_rate": successes / len(judged) if judged else None,
                           "judge_validation": "declared_by_input_report_not_independently_validated"},
            "duration": series([item["duration_s"] for item in trials], scope="trial_host_elapsed"),
            "metrics": [{"name": signature[0], **series(values, unit=signature[1], scope=signature[2])} for signature, values in sorted(metrics.items())]})
    statuses = {item["status"] for item in inputs}
    status = "FAIL" if statuses & {"FAIL", "FAULT"} else "BLOCKED" if "BLOCKED" in statuses else "STOPPED" if "STOPPED" in statuses else "PASS"
    return {"schema_version": 1, "scope": "verified_experiment_report_summary", "status": status,
            "inputs": inputs, "groups": result_groups,
            "population": {"valid_runs": len(populations), "requested_trials": sum(item["requested_repeats"] for item in populations),
                           "retained_trial_results": sum(len(item["trials"]) for item in populations),
                           "not_started_trials": sum(len(item["not_started_trials"]) for item in populations)},
            "not_verified": ["judge_correctness", "hardware_safety", "physical_calibration"]}
