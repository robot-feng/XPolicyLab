import asyncio
import unittest

import numpy as np

from client_server.ws import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit


class NestedImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_rpc_decodes_nested_batch_and_history(self):
        rgb = np.full((12, 16, 3), [220, 30, 10], dtype=np.uint8)
        bits = encode_image_bit(rgb)

        class ProbeModel:
            def begin_episode(self, obs):
                current = obs["batch"][0]
                history = current["history"][0]
                return {
                    "colors": [frame["vision"]["head"]["color"][0, 0]
                               for frame in [current, history]],
                    "depth": current["vision"]["head"]["depth"],
                    "indices": obs["indices"],
                }

        payload = {"batch": [{
            "vision": {"head": {"color": bits, "depth": b"untouched depth"}},
            "history": [{"vision": {"head": {"color": np.asarray(bytearray(bits), dtype=np.uint8)}}}],
        }], "indices": [0, 2]}
        server = PolicyServer(ProbeModel(), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()

        def call():
            client = WsModelClient(url=server.url, evaluation_id="nested-images", trial_id="episode-0")
            try:
                return client.call(func_name="begin_episode", obs=payload)
            finally:
                client.close()

        try:
            result = await asyncio.to_thread(call)
            np.testing.assert_allclose(result["colors"], [[220, 30, 10]] * 2, atol=3)
            self.assertEqual(result["depth"], b"untouched depth")
            self.assertEqual(result["indices"], [0, 2])
        finally:
            await server.stop()

    def test_aliases_and_cycles_do_not_redecode_images(self):
        rgb = np.full((12, 16, 3), [220, 30, 10], dtype=np.uint8)
        obs = {"vision": {"head": {"color": encode_image_bit(rgb)}}}
        obs["history"] = [obs]
        self.assertIs(decode_obs_images(obs), obs)
        np.testing.assert_allclose(obs["vision"]["head"]["color"][0, 0], [220, 30, 10], atol=3)
