import asyncio
from dataclasses import replace

import numpy as np
import pytest
from websockets.asyncio.client import connect

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.codec import decode_envelope, encode_frame
from client_server.ws.protocol.exceptions import ServerRestartedError, WsError
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
from xpolicy.core.config import ServerSettings
from xpolicy.core.contracts import ActionChunk
from xpolicy.policy.runtime import LegacyPolicyShim
from xpolicy.policy.serving.server import create_server


class LegacyModel:
    def __init__(self):
        self.updates = 0
        self.resets = 0
        self.ends = []
        self.observation = None

    def update_obs(self, obs):
        self.updates += 1
        self.observation = obs

    def get_action(self):
        return [{"joint_state": np.array([self.updates], dtype=np.float32)}]

    def reset(self):
        self.resets += 1

    def echo(self, obs):
        return obs

    def on_trial_end(self, result):
        self.ends.append(result)


def client_for(url):
    return PolicyEvalClient(PolicyEvalClientConfig(
        url=url, evaluation_id="cpu-regression", max_connect_attempts=1,
        connect_timeout_s=2, request_timeout_s=2, handshake_timeout_s=2,
        close_timeout_s=1, connect_retry_delay_s=0,
    ))


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=15))


def test_real_socket_full_legacy_rpc_and_dedupe(observation):
    async def scenario():
        model = LegacyModel()
        server = PolicyServer(LegacyPolicyShim(model), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        client = client_for(server.url)
        try:
            ack = await client.connect()
            assert ack.message_type == MessageType.HELLO_ACK
            assert ack.payload["server_instance_id"]
            assert (await client.reset(trial_id="trial-a")).message_type == MessageType.RESET_RESULT
            assert model.resets == 1
            wire = observation.to_wire_dict()
            bits = encode_image_bit(observation.images["cam_head"])
            wire["vision"]["cam_head"]["color"] = bits
            result = await client.infer(wire, trial_id="trial-a")
            assert result.message_type == MessageType.INFER_RESULT
            assert model.updates == 1
            np.testing.assert_array_equal(model.observation["vision"]["cam_head"]["color"], decode_image_bit(bits))
            echoed = await client.call("echo", wire)
            np.testing.assert_array_equal(echoed.payload["result"]["vision"]["cam_head"]["color"], decode_image_bit(bits))
            payload = {"observation": wire}
            first = await client.request(MessageType.INFER, payload, _request_id="same-request")
            second = await client.request(MessageType.INFER, payload, _request_id="same-request")
            assert first.payload["latency_ms"] == second.payload["latency_ms"]
            assert model.updates == 2
            with pytest.raises(WsError, match="already used"):
                await client.request(MessageType.RESET, {}, _request_id="same-request")
            with pytest.raises(WsError, match="no model method"):
                await client.call("predict_action_chunk", wire)
            result = await client.trial_end(trial_id="trial-a", result={"success": False})
            assert result.message_type == MessageType.TRIAL_END_ACK
            assert model.ends == [{"trial_id": "trial-a", "success": False}]
        finally:
            await client.close()
            await server.stop()
    run(scenario())


def test_chunk_service_preserves_rgb_context_and_disables_legacy(contracts, observation):
    async def scenario():
        server = create_server(contracts, ServerSettings(port=0))
        observed = []
        predict = server.model.model.predict
        def capture(obs):
            observed.append(obs)
            return predict(obs)
        server.model.model.predict = capture
        await server.start()
        client = client_for(server.url)
        try:
            await client.connect()
            wire = observation.to_wire_dict()
            bits = encode_image_bit(observation.images["cam_head"])
            wire["vision"]["cam_head"]["color"] = bits
            frame = await client.call("predict_action_chunk", wire)
            chunk = ActionChunk.from_wire_dict(frame.payload["result"])
            assert chunk.values.shape == (4, 2)
            assert (chunk.source_frame_id, chunk.episode_id, chunk.sequence_id) == (7, "episode-a", 0)
            assert chunk.clock_domain == "test_clock"
            assert server.model.model.calls == 1
            np.testing.assert_array_equal(observed[0].images["cam_head"], decode_image_bit(bits))
            assert observed[0].images["cam_head"][0, 0, 0] > observed[0].images["cam_head"][0, 0, 2]
            with pytest.raises(WsError, match="legacy INFER is disabled"):
                await client.infer(wire)
            with pytest.raises(WsError, match="no model method"):
                await client.call("get_action")
            assert server.model.model.calls == 1
            await client.reset(trial_id="trial-b")
            frame = await client.call("predict_action_chunk", replace(observation, episode_id="episode-b").to_wire_dict())
            assert frame.payload["result"]["sequence_id"] == 0
            assert frame.payload["result"]["episode_id"] == "episode-b"
            await client.trial_end(trial_id="trial-b")
        finally:
            await client.close()
            await server.stop()
    run(scenario())


def test_disconnect_retry_awaits_same_inflight_execution(observation):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        class SlowModel(LegacyModel):
            async def get_action(self):
                started.set()
                await release.wait()
                return super().get_action()
        model = SlowModel()
        server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        request = Frame(message_type=MessageType.INFER, request_id="retry-after-disconnect", evaluation_id="cpu-regression", payload={"observation": observation.to_wire_dict()})
        try:
            async with connect(server.url, compression=None) as first:
                await first.send(encode_frame(request))
                await asyncio.wait_for(started.wait(), timeout=2)
                await first.close()
            async with connect(server.url, compression=None) as second:
                await second.send(encode_frame(request))
                release.set()
                frame = decode_envelope(await asyncio.wait_for(second.recv(), timeout=2))
                assert frame.message_type == MessageType.INFER_RESULT
            assert model.updates == 1
        finally:
            release.set()
            await server.stop()
    run(scenario())


def test_restart_is_rejected_on_reconnect():
    async def scenario():
        server = PolicyServer(LegacyModel(), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        url = server.url
        port = server._server.sockets[0].getsockname()[1]
        client = client_for(url)
        replacement = None
        try:
            await client.connect()
            await client._reset_connection_state()
            await server.stop()
            replacement = PolicyServer(LegacyModel(), PolicyServerConfig(host="127.0.0.1", port=port))
            await replacement.start()
            with pytest.raises(ServerRestartedError):
                await client.connect()
            assert client._ws is None
        finally:
            await client.close()
            if replacement is not None:
                await replacement.stop()
            await server.stop()
    run(scenario())


def test_request_timeout_is_fatal_and_does_not_issue_fresh_inference(observation):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        class SlowModel(LegacyModel):
            async def get_action(self):
                started.set()
                await release.wait()
                return super().get_action()
        model = SlowModel()
        server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        client = client_for(server.url)
        try:
            await client.connect()
            with pytest.raises(WsError, match="timeout"):
                await client.request(MessageType.INFER, {"observation": observation.to_wire_dict()}, timeout_s=0.02)
            assert started.is_set()
            assert model.updates == 1
            assert not client._pending
        finally:
            release.set()
            await client.close()
            await server.stop()
    run(scenario())
