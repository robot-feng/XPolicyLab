"""Unified CLI for contract diagnosis and the existing policy RPC."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(prog="xpolicy", description="XPolicyLab unified contracts and policy service")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="validate declared policy/robot/dataset contracts")
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--mode", choices=["contract", "environment"], default="contract")
    doctor.add_argument("--verify-artifacts", action="store_true")
    doctor.add_argument("--output", type=Path)
    evaluation = commands.add_parser("eval", help="run dataset_direct replay with shared record-only execution")
    evaluation.add_argument("--config", type=Path, required=True)
    results = commands.add_parser("results", help="verify and summarize version 2 run reports")
    result_actions = results.add_subparsers(dest="results_action", required=True)
    summary = result_actions.add_parser("summarize", help="retain outcomes and pool comparable raw metric samples")
    summary.add_argument("reports", type=Path, nargs="+")
    summary.add_argument("--output", type=Path)
    assets = commands.add_parser("assets", help="inventory local source/asset attribution evidence without model imports")
    asset_actions = assets.add_subparsers(dest="assets_action", required=True)
    inventory = asset_actions.add_parser("inventory", help="scan tracked evidence and explicitly declared assets")
    inventory.add_argument("--root", type=Path, required=True)
    inventory.add_argument("--catalog", type=Path)
    inventory.add_argument("--output", type=Path, required=True)
    environments = commands.add_parser("environments", help="inspect environment implementation and execution capabilities")
    environment_actions = environments.add_subparsers(dest="environment_action", required=True)
    environment_list = environment_actions.add_parser("list", help="list providers and explicit deferred/unimplemented modes")
    environment_list.add_argument("--json", action="store_true")
    golden = commands.add_parser("golden", help="compare hash-verified diagnostic boundary samples")
    golden_actions = golden.add_subparsers(dest="golden_action", required=True)
    comparison = golden_actions.add_parser("diff", help="locate the first divergent boundary")
    comparison.add_argument("--expected", type=Path, required=True)
    comparison.add_argument("--observed", type=Path, required=True)
    comparison.add_argument("--layers", nargs="+", choices=["O1", "O2", "A0", "A1", "A2", "A3", "A4", "A5"])
    comparison.add_argument("--atol", type=float, default=0.0)
    comparison.add_argument("--rtol", type=float, default=0.0)
    comparison.add_argument("--output", type=Path)
    policy = commands.add_parser("policy", help="inspect or serve a policy")
    actions = policy.add_subparsers(dest="policy_action", required=True)
    listing = actions.add_parser("list", help="list factories without importing models")
    listing.add_argument("--json", action="store_true")
    service = actions.add_parser("serve", help="serve using XPolicyLab's WebSocket RPC")
    service.add_argument("--config", type=Path, required=True)
    service.add_argument("--host")
    service.add_argument("--port", type=int)
    return parser


async def _serve(server, api):
    try:
        await server.start()
        print(json.dumps({"event": "policy_server_ready", "url": server.url, "api": api}), flush=True)
        await server.serve_forever()
    finally:
        await server.stop()
        close = getattr(server.model, "close", None)
        if callable(close):
            close()


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from xpolicy.core.artifacts import write_json
            from xpolicy.diagnostics.doctor import doctor

            report = doctor(args.config, verify_artifacts=args.verify_artifacts, mode=args.mode)
            if args.output is not None:
                write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "PASS" else 2
        if args.command == "eval":
            import signal
            import threading
            from xpolicy.evaluation.runner import ExperimentRunner

            runner = ExperimentRunner(args.config)
            previous = None
            if threading.current_thread() is threading.main_thread():
                previous = signal.signal(signal.SIGINT, lambda signum, frame: runner.stop("interrupted"))
            try:
                report = runner.evaluate()
            finally:
                if previous is not None:
                    signal.signal(signal.SIGINT, previous)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "COMPLETED" else 2
        if args.command == "golden":
            from xpolicy.core.artifacts import write_json
            from xpolicy.diagnostics.golden import compare_bundles

            report = compare_bundles(args.expected, args.observed, layers=args.layers, atol=args.atol, rtol=args.rtol)
            if args.output is not None:
                write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "PASS" else 2
        if args.command == "results":
            from xpolicy.core.artifacts import write_json
            from xpolicy.evaluation.reports import summarize_reports

            report = summarize_reports(args.reports)
            if args.output is not None:
                write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "PASS" else 2
        if args.command == "assets":
            from xpolicy.core.artifacts import write_json
            from xpolicy.diagnostics.assets import inventory_assets

            if args.output.exists() or args.output.is_symlink():
                raise ValueError("asset inventory output must be a new file; retain previous audit reports")
            try:
                report = inventory_assets(args.root, args.catalog)
            except (OSError, ValueError) as exc:
                report = {"schema_version": 1, "scope": "local_asset_evidence_inventory", "status": "FAIL",
                          "root": str(args.root.resolve()), "error": {"type": type(exc).__name__, "message": str(exc)}}
            write_json(args.output, report)
            print(json.dumps({"report": str(args.output.resolve()), "status": report["status"], "summary": report.get("summary")}, ensure_ascii=False))
            return 0 if report["status"] == "DOCUMENTED" else 2
        if args.command == "environments":
            from xpolicy.environment.registry import environment_registry

            entries = environment_registry().entries()
            if args.json:
                print(json.dumps([asdict(entry) for entry in entries], indent=2))
            else:
                for entry in entries:
                    print(f"{entry.name:20s} {entry.status:16s} {entry.reason or ','.join(entry.capabilities)}")
            return 0
        if args.policy_action == "list":
            from xpolicy.core.registry import policy_registry

            entries = policy_registry().entries()
            if args.json:
                print(json.dumps([asdict(entry) for entry in entries], indent=2))
            else:
                for entry in entries:
                    print(f"{entry.name:24s} {','.join(entry.capabilities):16s} {entry.origin}")
            return 0
        from xpolicy.core.config import ServerSettings, load_service
        from xpolicy.policy.serving.server import create_server

        config, contracts = load_service(args.config)
        settings = config.server.model_dump()
        if args.host is not None:
            settings["host"] = args.host
        if args.port is not None:
            settings["port"] = args.port
        logging.basicConfig(level=logging.INFO)
        server = create_server(contracts, ServerSettings.model_validate(settings))
        asyncio.run(_serve(server, config.policy.api))
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"xpolicy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
