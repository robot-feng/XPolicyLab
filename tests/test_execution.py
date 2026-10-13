import asyncio
from dataclasses import replace

import numpy as np
import pytest

from xpolicy.core.config import ExecutionConfig
from xpolicy.core.contracts import ActionChunk, ActionType
from xpolicy.execution.async_latest import AsyncLatestOnly, WebSocketPolicyBackend
from xpolicy.execution.buffer import ActionBuffer
from xpolicy.execution.record_only import RecordOnlySink


def settings(**changes):
    return ExecutionConfig(control_hz=30, execute_horizon=3, **changes)


def prediction(ticket, values=None, **changes):
    return ActionChunk(values=np.arange(8, dtype=np.float32).reshape(4, 2) if values is None else values,
                       action_type="joint_position", joint_names=("arm_joint", "gripper"), action_units=("rad", "m"),
                       episode_id=ticket.episode_id, source_frame_id=ticket.source_frame_id,
                       sequence_id=ticket.sequence_id, clock_domain=ticket.source_clock_domain, **changes)


def buffer(contracts, **changes):
    events = []
    value = ActionBuffer(contracts.manifest, settings(**changes), emit=events.append)
    value.reset("episode-a")
    return value, events


def test_sync_owner_consumes_horizon_once(contracts, observation):
    value, events = buffer(contracts)
    request = value.issue(observation, control_tick=0)
    chunk = prediction(request)
    assert value.accept(request, chunk, control_tick=0)
    commands = [value.take(index) for index in range(3)]
    sink = RecordOnlySink()
    accepted = sink.submit_commands(commands)
    np.testing.assert_array_equal(accepted, chunk.values[:3])
    assert value.take(3) is None
    with pytest.raises(ValueError, match="only.*once"):
        value.take(3)
    with pytest.raises(ValueError, match="duplicate"):
        sink.submit_commands(commands)
    assert not accepted.flags.writeable
    assert [command.action_seq for command in commands] == [0, 1, 2]
    assert [item["trace_layer"] for item in events if "trace_layer" in item] == ["A1", "A2", "A3", "A3", "A3"]


def test_trim_uses_control_ticks_and_not_frame_ids(contracts, observation):
    value, events = buffer(contracts, latency_trim=True)
    request = value.issue(replace(observation, frame_id=90000), control_tick=0)
    chunk = prediction(request)
    assert value.accept(request, chunk, control_tick=2)
    np.testing.assert_array_equal(value.take(2).values, chunk.values[2])
    np.testing.assert_array_equal(value.take(3).values, chunk.values[3])
    assert value.take(4) is None
    assert next(item for item in events if item["stage"] == "execution_trim")["trimmed_steps"] == 2


def test_untrimmed_arrival_has_explicit_different_baseline(contracts, observation):
    value, _ = buffer(contracts)
    request = value.issue(observation, control_tick=0)
    chunk = prediction(request)
    value.accept(request, chunk, control_tick=2)
    np.testing.assert_array_equal(value.take(2).values, chunk.values[0])


def test_elapsed_horizon_and_already_taken_tick_are_not_replayed(contracts, observation):
    value, events = buffer(contracts, latency_trim=True)
    request = value.issue(observation, control_tick=0)
    assert value.take(0) is None
    value.accept(request, prediction(request), control_tick=0)
    command = value.take(1)
    assert command.contributors[0].predicted_step_index == 1
    late = value.issue(replace(observation, frame_id=8), control_tick=1)
    assert not value.accept(late, prediction(late), control_tick=5)
    assert value.pending_count == 0
    assert events[-1]["reason"] == "prediction_horizon_elapsed"


def test_reset_same_episode_discards_previous_generation(contracts, observation):
    value, events = buffer(contracts)
    old = value.issue(observation, control_tick=0)
    value.accept(old, prediction(old), control_tick=0)
    stale = value.issue(replace(observation, frame_id=8), control_tick=1)
    generation = value.reset("episode-a")
    assert generation == old.generation + 1
    assert value.pending_count == 0
    assert not value.accept(stale, "malformed obsolete result", control_tick=0)
    assert events[-1]["reason"] == "old_generation"


def test_duplicate_and_out_of_order_result_rejected(contracts, observation):
    value, events = buffer(contracts)
    first = value.issue(observation, control_tick=0)
    second = value.issue(replace(observation, frame_id=8), control_tick=1)
    assert value.accept(second, prediction(second), control_tick=1)
    assert not value.accept(first, prediction(first), control_tick=1)
    assert events[-1]["reason"] == "out_of_order_result"
    assert not value.accept(second, prediction(second), control_tick=1)
    assert events[-1]["reason"] == "duplicate_or_unknown_request"


@pytest.mark.parametrize("changes", [{"episode_id": "other"}, {"source_frame_id": 999}, {"sequence_id": 4}, {"clock_domain": "other_clock"}, {"joint_names": ("gripper", "arm_joint")}, {"action_units": ("deg", "m")}])
def test_prediction_semantics_and_context_checked(contracts, observation, changes):
    value, _ = buffer(contracts)
    request = value.issue(observation, control_tick=0)
    bad = replace(prediction(request), **changes)
    with pytest.raises(ValueError, match="prediction"):
        value.accept(request, bad, control_tick=0)
    assert value.pending_count == 0


def test_deadline_clock_is_checked_on_accept_and_consumption(contracts, observation):
    value, events = buffer(contracts)
    request = value.issue(observation, control_tick=0)
    chunk = prediction(request, valid_until_s=3)
    assert not value.accept(request, chunk, control_tick=0, now_s=1.5, clock_domain="host_monotonic")
    assert events[-1]["reason"] == "unverified_deadline_clock"
    request = value.issue(observation, control_tick=0)
    chunk = prediction(request, valid_until_s=3)
    assert value.accept(request, chunk, control_tick=0, now_s=1.5, clock_domain="test_clock")
    assert value.take(0, now_s=3, clock_domain="test_clock") is None
    assert events[-1]["reason"] == "expired_deadline"


def test_absolute_overlap_fixed_weights_and_provenance(contracts, observation):
    value, _ = buffer(contracts, overlap_blend_steps=2)
    old = value.issue(observation, control_tick=0)
    value.accept(old, prediction(old, np.full((4, 2), 10, dtype=np.float32)), control_tick=0)
    newer = value.issue(replace(observation, frame_id=8), control_tick=0)
    value.accept(newer, prediction(newer, np.full((4, 2), 100, dtype=np.float32)), control_tick=0)
    first, second, third = [value.take(index) for index in range(3)]
    np.testing.assert_allclose(first.values, [40, 40])
    np.testing.assert_allclose(second.values, [70, 70])
    np.testing.assert_allclose(third.values, [100, 100])
    assert first.contributors[0].ticket == old
    assert first.contributors[1].ticket == newer
    assert [item.weight for item in first.contributors] == pytest.approx([2 / 3, 1 / 3])
    assert sum(item.weight for item in second.contributors) == pytest.approx(1)


def test_expired_old_plan_is_not_blended_and_old_tail_cannot_reappear(contracts, observation):
    value, _ = buffer(contracts, overlap_blend_steps=2, latency_trim=True)
    first = value.issue(observation, control_tick=0)
    value.accept(first, prediction(first, valid_until_s=1.6), control_tick=0, now_s=1.5, clock_domain="test_clock")
    newer = value.issue(replace(observation, frame_id=8), control_tick=0)
    value.accept(newer, prediction(newer, np.full((4, 2), 100, dtype=np.float32)), control_tick=0, now_s=2, clock_domain="test_clock")
    command = value.take(0)
    np.testing.assert_array_equal(command.values, [100, 100])
    assert len(command.contributors) == 1
    newest = value.issue(replace(observation, frame_id=9), control_tick=0)
    value.accept(newest, prediction(newest, np.full((4, 2), 200, dtype=np.float32)), control_tick=3)
    np.testing.assert_array_equal(value.take(3).values, [200, 200])
    assert value.take(4) is None


def test_smoothing_other_action_spaces_fails_before_inference(contracts):
    manifest = contracts.manifest.model_copy(update={"action_type": ActionType.JOINT_DELTA})
    with pytest.raises(ValueError, match="absolute joint_position"):
        ActionBuffer(manifest, settings(overlap_blend_steps=1))


def test_invalid_reset_does_not_change_active_generation(contracts, observation):
    value, _ = buffer(contracts)
    request = value.issue(observation, control_tick=0)
    value.accept(request, prediction(request), control_tick=0)
    generation = value.generation
    with pytest.raises(ValueError, match="nonnegative integer"):
        value.reset("episode-b", start_tick=True)
    assert value.generation == generation
    assert value.episode_id == "episode-a"
    assert value.take(0) is not None


class ControlledBackend:
    def __init__(self):
        self.resets, self.started, self.releases = [], asyncio.Queue(), asyncio.Queue()
        self.calls, self.sequence = 0, 0

    async def reset(self, episode_id):
        self.resets.append(episode_id)
        self.sequence = 0

    async def predict(self, obs, request):
        self.calls += 1
        sequence = self.sequence
        self.sequence += 1
        await self.started.put(obs)
        result = await self.releases.get()
        if isinstance(result, Exception):
            raise result
        return ActionChunk(values=np.full((4, 2), obs.frame_id, dtype=np.float32), action_type="joint_position",
                           joint_names=("arm_joint", "gripper"), action_units=("rad", "m"), episode_id=obs.episode_id,
                           source_frame_id=obs.frame_id, sequence_id=sequence, clock_domain=obs.clock_domain)


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=10))


def test_async_reset_is_immediate_and_model_calls_remain_serial(contracts, observation):
    async def scenario():
        backend, events = ControlledBackend(), []
        controller = AsyncLatestOnly(backend, contracts, settings(strategy="async_latest_only", latency_trim=True), emit=events.append)
        controller.reset("episode-a")
        controller.offer(observation, control_tick=0)
        assert (await backend.started.get()).episode_id == "episode-a"
        previous_generation = controller.buffer.generation
        assert controller.reset("episode-b") == previous_generation + 1
        assert controller.take(0) is None
        new_obs = replace(observation, episode_id="episode-b", frame_id=0)
        controller.offer(new_obs, control_tick=1)
        assert backend.resets == ["episode-a"]
        assert backend.calls == 1
        await backend.releases.put(True)
        assert await backend.started.get() == new_obs
        assert backend.resets == ["episode-a", "episode-b"]
        await backend.releases.put(True)
        await controller.wait_idle()
        assert controller.buffer.pending_count > 0
        assert any(item.get("reason") == "old_generation" for item in events)
        command = controller.take(2)
        assert command.ticket.generation == previous_generation + 1
        assert command.ticket.episode_id == "episode-b"
        assert await controller.close()
    run(scenario())


def test_async_latest_only_bounds_the_pending_observation(contracts, observation):
    async def scenario():
        backend = ControlledBackend()
        controller = AsyncLatestOnly(backend, contracts, settings(strategy="async_latest_only", latency_trim=True))
        controller.reset("episode-a")
        controller.offer(observation, control_tick=0)
        await backend.started.get()
        for index in range(1, 4):
            controller.offer(replace(observation, frame_id=7 + index), control_tick=index)
        assert backend.calls == 1
        assert controller.superseded_observations == 2
        await backend.releases.put(True)
        assert (await backend.started.get()).frame_id == 10
        await backend.releases.put(True)
        await controller.wait_idle()
        assert backend.calls == 2
        assert controller.buffer.pending_count <= 3
        assert await controller.close()
    run(scenario())


def test_timeout_stops_consumption_without_canceling_a_running_call(contracts, observation):
    async def scenario():
        backend = ControlledBackend()
        controller = AsyncLatestOnly(backend, contracts, settings(strategy="async_latest_only"))
        controller.reset("episode-a")
        controller.offer(observation, control_tick=0)
        await backend.started.get()
        with pytest.raises(TimeoutError):
            await controller.wait_idle(0.01)
        assert controller.fault is not None
        assert controller.take(0) is None
        assert not controller._worker.done()
        await backend.releases.put(True)
        await controller.wait_idle()
        assert controller.buffer.pending_count == 0
        assert await controller.close()
    run(scenario())


def test_obsolete_error_does_not_poison_new_episode(contracts, observation):
    async def scenario():
        backend = ControlledBackend()
        controller = AsyncLatestOnly(backend, contracts, settings(strategy="async_latest_only"))
        controller.reset("episode-a")
        controller.offer(observation, control_tick=0)
        await backend.started.get()
        controller.reset("episode-b")
        controller.offer(replace(observation, episode_id="episode-b"), control_tick=0)
        await backend.releases.put(RuntimeError("old inference failed"))
        await backend.started.get()
        await backend.releases.put(True)
        await controller.wait_idle()
        assert controller.fault is None
        assert controller.take(0) is not None
        await controller.close()
    run(scenario())


def test_actual_websocket_async_reset_discards_inflight_old_generation(contracts, observation):
    from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
    from xpolicy.core.config import ServerSettings
    from xpolicy.policy.serving.server import create_server

    async def scenario():
        server = create_server(contracts, ServerSettings(port=0))
        started, releases, events = asyncio.Queue(), asyncio.Queue(), []
        original = server.model.predict_action_chunk
        calls = []
        async def delayed(wire):
            calls.append(wire["episode_id"])
            await started.put(wire["episode_id"])
            await releases.get()
            return original(wire)
        server.model.predict_action_chunk = delayed
        await server.start()
        client = PolicyEvalClient(PolicyEvalClientConfig(url=server.url, evaluation_id="async-reset-regression", request_timeout_s=2,
                                                       connect_timeout_s=2, handshake_timeout_s=2, max_connect_attempts=1, close_timeout_s=1))
        controller = AsyncLatestOnly(WebSocketPolicyBackend(client), contracts,
                                     settings(strategy="async_latest_only", latency_trim=True), emit=events.append)
        try:
            await client.connect()
            controller.reset("episode-a")
            controller.offer(observation, control_tick=0)
            assert await started.get() == "episode-a"
            old_generation = controller.buffer.generation
            controller.reset("episode-b")
            new_obs = replace(observation, episode_id="episode-b", frame_id=99, state=np.array([0.8, 0.04], dtype=np.float32))
            assert controller.take(0) is None
            controller.offer(new_obs, control_tick=0)
            assert calls == ["episode-a"]
            await releases.put(True)
            assert await started.get() == "episode-b"
            await releases.put(True)
            await controller.wait_idle(2)
            command = controller.take(1)
            assert command.ticket.generation == old_generation + 1
            assert command.ticket.source_frame_id == 99
            assert command.ticket.sequence_id == 0
            assert command.ticket.request_id in server._response_cache
            np.testing.assert_array_equal(command.values, new_obs.state)
            assert any(item.get("reason") == "old_generation" for item in events)
            assert not any(item["stage"] == "command" and item["episode_generation"] == old_generation for item in events)
            assert calls == ["episode-a", "episode-b"]
        finally:
            await releases.put(True)
            await controller.close(timeout_s=2)
            await client.close()
            await server.stop()
    run(scenario())
