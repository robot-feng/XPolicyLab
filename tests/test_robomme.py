import asyncio
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model, encode_obs, encode_robomme_obs
from XPolicyLab.utils.process_data import unpack_robot_state
from XPolicyLab.utils.robomme_env_client import joint_action, observations, run_episode
from XPolicyLab.utils import robomme_env_client
from client_server.ws.model_server import PolicyServer, PolicyServerConfig

PANDA = {"arm_dim": [7], "ee_dim": [1]}
DUAL = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def raw_observation(count=1):
    return {
        "front_rgb_list": [np.full((8, 10, 3), [220, 30, 10], np.uint8) for _ in range(count)],
        "wrist_rgb_list": [np.full((8, 10, 3), [10, 40, 230], np.uint8) for _ in range(count)],
        "joint_state_list": [np.arange(7, dtype=np.float32) for _ in range(count)],
        "gripper_state_list": [np.array([0.03, 0.04], np.float32) for _ in range(count)],
    }


class RoboMMETests(unittest.TestCase):
    def test_panda_observation_preserves_rgb_and_physical_finger_units(self):
        obs = list(observations(raw_observation(3), ["pick twice"]))
        self.assertEqual(len(obs), 3)
        encoded = encode_robomme_obs(obs[0], PANDA)
        self.assertEqual(encoded["prompt"], "pick twice")
        np.testing.assert_array_equal(encoded["observation/image"][:, 0, 0], [220, 30, 10])
        np.testing.assert_array_equal(encoded["observation/wrist_image"][:, 0, 0], [10, 40, 230])
        np.testing.assert_allclose(encoded["observation/state"], [0, 1, 2, 3, 4, 5, 6, 0.03])
        raw = raw_observation(2)
        raw["wrist_rgb_list"].pop()
        with self.assertRaisesRegex(ValueError, "aligned"):
            list(observations(raw, "pick"))

    def test_joint_action_round_trip_and_invalid_output_rejected(self):
        vector = np.r_[np.arange(7), -1].astype(np.float32)
        action = unpack_robot_state(vector, "joint", PANDA)
        np.testing.assert_array_equal(joint_action(action, PANDA), vector)
        action["ee_joint_state"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            joint_action(action, PANDA)
        with self.assertRaises(KeyError):
            joint_action({"arm_joint_state": vector[:7]}, PANDA)

    def test_original_dual_arm_camera_and_state_contract_is_preserved(self):
        vision = {name: {"color": np.full((8, 10, 3), value, np.uint8)} for name, value in
                  [("cam_head", 10), ("cam_left_wrist", 30), ("cam_right_wrist", 50)]}
        obs = {"vision": vision, "state": unpack_robot_state(np.arange(14), "joint", DUAL),
               "instruction": "stack"}
        encoded = encode_obs(obs, "joint", DUAL)
        np.testing.assert_array_equal(encoded["state"], np.arange(14))
        self.assertEqual(set(encoded["images"]), {"cam_high", "cam_left_wrist", "cam_right_wrist"})
        self.assertEqual(encoded["images"]["cam_right_wrist"].shape, (3, 8, 10))

    def test_pi05_subset_order_empty_batch_and_reset(self):
        class Policy:
            resets = 0
            def infer(self, obs):
                return {"actions": np.tile(obs["observation/state"], (20, 1))}
            def reset(self):
                self.resets += 1

        policy = Policy()
        with patch.object(Model, "get_model", return_value=policy), patch(
                "XPolicyLab.policy.Pi_05.model.get_robot_action_dim_info", return_value=PANDA):
            model = Model({"task_name": "PickXtimes", "bench_name": "RoboMME", "env_cfg_type": "panda"})
        obs = list(observations(raw_observation(2), "pick"))
        obs[0]["env_idx"], obs[1]["env_idx"] = 8, 3
        obs[1]["state"]["arm_joint_state"] = np.full(7, 0.5)
        model.update_obs_batch(obs)
        actions = model.get_action_batch([3, 8])
        np.testing.assert_array_equal(actions[0][0]["arm_joint_state"], np.full(7, 0.5))
        self.assertEqual(len(actions[1]), 20)
        self.assertEqual(model.get_action_batch([]), [])
        model.reset()
        self.assertEqual(policy.resets, 1)
        with self.assertRaises(AssertionError):
            model.get_action()
        with self.assertRaisesRegex(ValueError, "panda"):
            Model({"task_name": "x", "bench_name": "RoboMME", "env_cfg_type": "arx_x5"})

    def test_loop_forwards_demos_and_respects_chunk_limit_and_native_termination(self):
        class Env:
            steps = 0
            def reset(self):  # Calling with a user-supplied seed would fail this contract.
                return raw_observation(3), {"task_goal": ["pick"]}
            def step(self, action):
                np.testing.assert_array_equal(action, np.r_[np.zeros(7), 1])
                self.steps += 1
                return raw_observation(), 0, False, self.steps == 5, {"status": "timeout", "task_goal": ["pick"]}

        class Client:
            resets = updates = calls = 0
            def call(self, func_name, obs=None):
                if func_name == "reset":
                    self.resets += 1
                elif func_name.startswith("update_obs"):
                    self.updates += 1
                else:
                    self.calls += 1
                    chunk = unpack_robot_state(np.tile(np.r_[np.zeros(7), 1], (20, 1)), "joint", PANDA)
                    return [chunk] if func_name.endswith("batch") else chunk

        for batch in (True, False):
            client = Client()
            result = run_episode(Env(), client, PANDA, action_steps=2, eval_batch=batch)
            self.assertEqual((client.resets, client.updates, client.calls), (1, 8, 3))
            self.assertEqual((result["steps"], result["initial_frames"], result["status"]), (5, 3, "timeout"))
            self.assertFalse(result["success"])


class RoboMMEEntryPointTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_uses_real_websocket_client_and_writes_episode_report(self):
        class Policy:
            def reset(self):
                pass
            def update_obs_batch(self, obs):
                self.obs = obs[0]
            def get_action_batch(self, env_idx_list=None):
                return [[self.obs["state"]]]

        class Env:
            closed = False
            def reset(self):
                return raw_observation(), {"task_goal": ["pick"]}
            def step(self, action):
                return raw_observation(), 0, False, True, {"task_goal": ["pick"], "status": "timeout"}
            def close(self):
                self.closed = True

        env = Env()
        class Builder:
            def __init__(self, **kwargs):
                pass
            def get_episode_num(self):
                return 50
            def resolve_episode(self, episode):
                return 123, "easy"
            def make_env_for_episode(self, episode):
                return env

        server = PolicyServer(Policy(), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                wrapper = types.ModuleType("robomme.env_record_wrapper")
                wrapper.__file__ = str(root / "__init__.py")
                wrapper.BenchmarkEnvBuilder = Builder
                for file in ("episode_config_resolver.py", "DemonstrationWrapper.py", "RecordWrapper.py"):
                    (root / file).write_text("# test fixture\n")
                modules = {"robomme": types.ModuleType("robomme"), "robomme.env_record_wrapper": wrapper}
                output = root / "results"
                port = server.url.rsplit(":", 1)[1]
                argv = ["robomme_env_client", "--task", "PickXtimes", "--host", "127.0.0.1", "--port", port,
                        "--output", str(output), "--no-video"]
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), patch.object(
                        robomme_env_client, "get_robot_action_dim_info", return_value=PANDA), patch(
                        "importlib.metadata.version", return_value="test-fixture"):
                    await asyncio.to_thread(robomme_env_client.main)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "COMPLETED")
                self.assertEqual(report["completed"], 1)
                self.assertEqual(report["episodes"][0]["benchmark_seed"], 123)
                self.assertEqual(report["episodes"][0]["steps"], 1)
                self.assertEqual(report["success_rate"], 0)
                self.assertTrue(env.closed)
        finally:
            await server.stop()


if __name__ == "__main__":
    unittest.main()
