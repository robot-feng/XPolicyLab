"""Supervise one owned process group without touching unrelated processes."""

from __future__ import annotations

import os
import math
from pathlib import Path
import signal
import subprocess
import time


def run_owned_process(command, cwd, environment, timeout_s, log_path, *, stop_signal=None,
                      interrupt_first=False, grace_s=5., on_started=None):
    """Retain exits and attempt bounded cleanup, including orphan SDK workers.

    A process deadline is independent of a Runner's per-trial budget. A root
    surviving KILL remains explicitly unreaped; callers must stop scheduling.
    No device stop or physical halt is implied by process termination.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0 or isinstance(grace_s, bool) or not math.isfinite(grace_s) or not 0 < grace_s <= 30:
        raise ValueError("process timeout must be finite and positive, and cleanup grace in (0, 30]")
    if not hasattr(os, "killpg"):
        raise RuntimeError("owned process group supervision requires a POSIX runtime")
    started = time.monotonic()
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        termination_signals, reason = [], None

        def terminate(signals):
            for sig in signals:
                try:
                    os.killpg(process.pid, sig)
                    termination_signals.append(sig.name)
                except ProcessLookupError:
                    return
                try:
                    process.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    pass
                # Continue even when the root exits: its workers may ignore TERM.

        escalation = ([signal.SIGINT] if interrupt_first else []) + [signal.SIGTERM, signal.SIGKILL]
        try:
            if on_started is not None:
                on_started(process.pid)
            while process.poll() is None:
                request = stop_signal.request_info if stop_signal is not None else None
                if request is not None:
                    reason = request["reason"]
                    break
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    reason = "process_timeout"
                    break
                try:
                    process.wait(timeout=min(.05, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if reason is not None:
                terminate(escalation)
            else:
                # Retain the root's original exit; clean up only surviving workers.
                terminate([signal.SIGTERM, signal.SIGKILL])
        except BaseException:
            terminate(escalation)
            raise
    return {"command": command, "cwd": str(cwd), "pid": process.pid, "exit_code": process.poll(),
            "root_reaped": process.returncode is not None, "termination_signals": termination_signals,
            "stop_reason": reason, "timed_out": reason == "process_timeout",
            "timeout_s": timeout_s, "cleanup_grace_s": grace_s,
            "duration_s": time.monotonic() - started}
