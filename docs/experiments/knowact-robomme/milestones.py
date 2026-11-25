"""Run four-GPU KnowAct training and fixed RoboMME evaluations in alternating stages."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import subprocess

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POLICY = REPO / "policy/KnowAct"
TASKS = ("MoveCube", "PatternLock", "RouteStick", "InsertPeg")


def check(command, **kwargs):
    subprocess.run(command, cwd=REPO, check=True, **kwargs)


def checkpoint_step(path):
    if not path.exists():
        return 0
    return int(
        torch.load(path, weights_only=True, map_location="cpu", mmap=True)["step"]
    )


def evaluate(path, split, task, episode, folder, args):
    # Completed trials are reusable when a driver is restarted. Preserve failed attempts.
    for attempt in range(1000):
        out = folder / f"{split}-{task}-{episode}-{attempt}"
        report = out / "report.json"
        if report.is_file():
            result = json.loads(report.read_text())
            if result["status"] == "completed" and result["completed"] == 1:
                return result["episodes"][0]
        if not out.exists():
            break
    env = dict(
        os.environ,
        EVAL_ENV_TYPE="sim",
        KNOWACT_CKPT_PATH=str(path),
        ROBOMME_OUTPUT_DIR=str(out),
        ROBOMME_SPLIT=split,
        ROBOMME_START_EPISODE=str(episode),
        ROBOMME_EPISODES="1",
        ROBOMME_MAX_STEPS="1300",
        ROBOMME_ACTION_STEPS="16",
        KNOWACT_CONTEXT_MAX_TOKENS="65536",
    )
    with out.with_suffix(".log").open("a") as log:
        check(
            [
                "bash",
                POLICY / "eval.sh",
                "RoboMME",
                task,
                args.name,
                "panda",
                "joint",
                "7",
                "2",
                "2",
                os.environ["KNOWACT_PYTHON"],
                args.sim_env,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = json.loads(report.read_text())
    if result["status"] != "completed" or result["completed"] != 1:
        raise RuntimeError(f"Incomplete RoboMME trial in {out}")
    return result["episodes"][0]


def trial(path, split, task, ep, folder, args):
    for attempt in range(2):
        try:
            return evaluate(path, split, task, ep, folder, args)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"{task} {split}/{ep} attempt {attempt + 1}: {exc}", flush=True)
    return dict(episode=ep, status="evaluation_error", logs=str(folder))


def main(args):
    root = Path(os.environ["KNOWACT_OUTPUT_ROOT"]).resolve()
    run = root / f"RoboMME-{args.name}-panda-joint-7"
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "milestones.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_pid:
        proc = Path(f"/proc/{args.wait_pid}/stat")

        def identity():
            try:
                fields = proc.read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                return None
            return fields[19] if fields[0] != "Z" else None

        started = identity()
        while started and identity() == started:
            time.sleep(5)
    for step in (10000, 20000, 30000):
        last = checkpoint_step(run / "last.pt")
        if last < step:
            resume = ["--resume"] if last else []
            check(
                [
                    "bash",
                    POLICY / "train.sh",
                    "RoboMME",
                    args.name,
                    "panda",
                    "joint",
                    "7",
                    "0,1,2,3",
                    "--stop-at",
                    str(step),
                    *resume,
                ]
            )
        numbered = run / "checkpoints" / f"step_{step:06d}.pt"
        if checkpoint_step(numbered) != step:
            raise RuntimeError(f"Missing milestone {numbered}")
        sources = {"last": numbered}
        if step == 30000 and checkpoint_step(run / "best.pt") != step:
            sources["best"] = run / "best.pt"
        split, tasks, episodes = (
            ("test", TASKS, (0, 2, 3)) if step == 30000 else ("val", TASKS[:1], (0,))
        )
        results = {}
        for label, path in sources.items():
            folder = run / "rollouts" / f"{step}-{label}"
            folder.mkdir(parents=True, exist_ok=True)
            results[label] = {
                task: [trial(path, split, task, ep, folder, args) for ep in episodes]
                for task in tasks
            }
        metric = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        payload = dict(
            step=step,
            run=args.name,
            split=split,
            context_token_limit=65536,
            holdout_ce=next(row["holdout_ce"] for row in metric if row["step"] == step),
            checkpoints={key: checkpoint_step(path) for key, path in sources.items()},
            results=results,
        )
        dest = HERE / f"progress-{args.name}-{step}.json"
        dest.write_text(json.dumps(payload, indent=2) + "\n")
        if args.sync_branch:
            try:
                publish(dest, args.sync_branch, step)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"Milestone report saved locally at {dest}: {exc}", flush=True)


def publish(dest, branch, step):
    current = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    if current != branch:
        raise RuntimeError("Checkout branch changed")
    check(["git", "add", "--", dest])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", dest], cwd=REPO)
    if diff.returncode == 1:
        check(
            [
                "git",
                "commit",
                "--only",
                "-m",
                f"Record KnowAct RoboMME step {step}",
                "--",
                dest,
            ]
        )
    else:
        diff.check_returncode()
    check(["git", "push", "origin", branch])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="knowact08-state")
    parser.add_argument(
        "--sim-env", default=str(REPO.parent / "robomme_benchmark/.venv")
    )
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--sync-branch")
    main(parser.parse_args())
