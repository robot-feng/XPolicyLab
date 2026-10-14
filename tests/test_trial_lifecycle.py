import io
import asyncio
import threading

import pytest

from xpolicy.evaluation.lifecycle import StopSignal, TrialLifecycle, TrialState
from xpolicy.evaluation.metrics import series


def test_lifecycle_transitions_stop_and_finalization_are_owned_once():
    events = []
    lifecycle = TrialLifecycle(lambda stage, **event: events.append((stage, event)))
    with pytest.raises(RuntimeError, match="illegal"):
        lifecycle.transition(TrialState.RUNNING)
    for state in (TrialState.PREFLIGHT, TrialState.PREPARED, TrialState.READY, TrialState.RUNNING):
        lifecycle.transition(state)
    assert lifecycle.stop("operator_stop")
    assert not lifecycle.stop("fault")
    lifecycle.finalize()
    lifecycle.finalize(failed=True)
    assert lifecycle.report()["state"] == "COMPLETED"
    assert lifecycle.report()["stop_reason"] == "operator_stop"
    assert lifecycle.report()["device_stop_ack"] is None
    assert sum(stage == "trial_stop" for stage, _ in events) == 1
    with pytest.raises(RuntimeError):
        lifecycle.transition(TrialState.RUNNING)


def test_stop_signal_is_idempotent_and_returns_an_owned_snapshot():
    stop = StopSignal()
    assert stop.request_info is None
    assert stop.request()
    assert not stop.request("interrupted")
    snapshot = stop.request_info
    snapshot["reason"] = "changed"
    assert stop.request_info["reason"] == "operator_stop"
    with pytest.raises(ValueError):
        stop.request("reset_robot")


def test_reentrant_cli_stop_request_cannot_deadlock_on_main_thread():
    signal = StopSignal()
    with signal._lock:
        assert signal.request("interrupted")
        assert signal.request_info["reason"] == "interrupted"


def test_metric_samples_have_explicit_empty_and_quantile_semantics():
    assert series([])["p95"] is None
    assert series([])["status"] == "NOT_MEASURED"
    metric = series([1., 2., 3., 4.])
    assert metric["p50"] == 2.5
    assert metric["p95"] == pytest.approx(3.85)
    assert metric["samples"] == 4
    assert series([1e308, 1e308])["p99"] == 1e308


@pytest.mark.parametrize("samples", [[True], ["1"], [-1], [float("nan")], [float("inf")]])
def test_invalid_metric_sample_rejected(samples):
    with pytest.raises(ValueError):
        series(samples)


def test_async_stop_waiter_is_woken_from_another_thread_and_unregisters():
    async def run():
        signal = StopSignal()
        assert not await signal.wait_async(0.001)
        waiter = asyncio.create_task(signal.wait_async(1))
        await asyncio.sleep(0)
        thread = threading.Thread(target=signal.request)
        thread.start(); thread.join()
        assert await waiter
        assert signal._listeners == set()
        assert await signal.wait_async(1)
    asyncio.run(run())
