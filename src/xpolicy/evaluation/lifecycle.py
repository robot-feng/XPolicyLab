"""Runner-owned trial transitions and a thread-safe, idempotent stop request."""

from enum import Enum
import asyncio
import threading
import time


class TrialState(str, Enum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    PREPARED = "PREPARED"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


TERMINAL = {TrialState.COMPLETED, TrialState.FAILED, TrialState.BLOCKED}
NEXT = {
    TrialState.CREATED: {TrialState.PREFLIGHT, TrialState.STOPPING},
    TrialState.PREFLIGHT: {TrialState.PREPARED, TrialState.BLOCKED, TrialState.STOPPING},
    TrialState.PREPARED: {TrialState.READY, TrialState.BLOCKED, TrialState.STOPPING},
    TrialState.READY: {TrialState.RUNNING, TrialState.STOPPING},
    TrialState.RUNNING: {TrialState.STOPPING},
    TrialState.STOPPING: {TrialState.FINALIZING},
    TrialState.FINALIZING: {TrialState.COMPLETED, TrialState.FAILED},
}


class StopSignal:
    """Requests host-side stopping; it is not a device stop or device ACK."""

    def __init__(self):
        # A CLI signal handler may request stopping while the main thread is
        # reading this same object. Reentrancy prevents a signal-time deadlock.
        self._lock = threading.RLock()
        self._request = None
        self._listeners = set()

    def request(self, reason="operator_stop") -> bool:
        if reason not in {"operator_stop", "interrupted"}:
            raise ValueError("stop reason must be operator_stop or interrupted")
        with self._lock:
            if self._request is not None:
                return False
            self._request = {"reason": reason, "monotonic_s": time.monotonic()}
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener()
        return True

    async def wait_async(self, timeout_s):
        loop, event = asyncio.get_running_loop(), asyncio.Event()

        def wake():
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # The request remains recorded even if its waiter just closed.
                pass

        with self._lock:
            self._listeners.add(wake)
            already_requested = self._request is not None
        try:
            if already_requested:
                event.set()
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            with self._lock:
                self._listeners.discard(wake)

    @property
    def request_info(self):
        with self._lock:
            return None if self._request is None else dict(self._request)


class StopRequested(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class TrialLifecycle:
    def __init__(self, emit, *, clock=time.monotonic):
        self.emit, self.clock = emit, clock
        self.state, self.stop_reason, self.started_s = TrialState.CREATED, None, None
        self.history = []
        self._record(None)

    def _record(self, previous):
        item = {"from_state": previous.value if previous is not None else None,
                "state": self.state.value, "monotonic_s": self.clock()}
        self.history.append(item)
        self.emit("trial_state", **item)

    def transition(self, state):
        state = TrialState(state)
        if state not in NEXT.get(self.state, set()):
            raise RuntimeError(f"illegal trial transition: {self.state.value} -> {state.value}")
        previous, self.state = self.state, state
        if state == TrialState.RUNNING:
            self.started_s = self.clock()
        self._record(previous)

    def stop(self, reason):
        if reason not in {"dataset_end", "timeout", "operator_stop", "interrupted", "fault"}:
            raise ValueError("unknown trial stop reason")
        if self.state in TERMINAL or self.stop_reason is not None:
            return False
        self.stop_reason = reason
        self.transition(TrialState.STOPPING)
        self.emit("trial_stop", reason=reason, scope="host_record_only", device_ack=None)
        return True

    def finalize(self, *, failed=False):
        if self.state in TERMINAL:
            return
        if self.stop_reason is None:
            raise RuntimeError("trial must stop before finalization")
        self.transition(TrialState.FINALIZING)
        self.transition(TrialState.FAILED if failed else TrialState.COMPLETED)

    def report(self):
        return {"state": self.state.value, "stop_reason": self.stop_reason,
                "started": self.started_s is not None, "history": list(self.history),
                "device_stop_ack": None}
