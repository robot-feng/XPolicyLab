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
    doctor.add_argument("--mode", choices=["contract"], default="contract")
    doctor.add_argument("--verify-artifacts", action="store_true")
    doctor.add_argument("--output", type=Path)
    evaluation = commands.add_parser("eval", help="run synchronous dataset_direct replay into a record-only sink")
    evaluation.add_argument("--config", type=Path, required=True)
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

            report = doctor(args.config, verify_artifacts=args.verify_artifacts)
            if args.output is not None:
                write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "PASS" else 2
        if args.command == "eval":
            from xpolicy.evaluation.runner import run_experiment

            report = run_experiment(args.config)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if report["status"] == "COMPLETED" else 2
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
