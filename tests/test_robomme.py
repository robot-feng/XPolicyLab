import unittest

import numpy as np

from XPolicyLab.policy.demo_policy.deploy import eval_one_episode, eval_one_episode_batch
from XPolicyLab.benchmarks.robomme.client import Episode, EpisodeRPC, observations


def frames(indices):
    rgb = [np.full((4, 5, 3), [i, 4, 9], np.uint8) for i in indices]
    return {"front_rgb_list": rgb, "wrist_rgb_list": [frame + 1 for frame in rgb],
            "joint_state_list": [np.arange(7) for _ in rgb],
            "gripper_state_list": [np.array([0.03, 0.04]) for _ in rgb]}


class RoboMMETests(unittest.TestCase):
    def test_existing_policy_deploy_loops_replay_demo_and_stop_on_native_termination(self):
        class Simulator:
            def reset(self):
                self.actions = []
                return frames([0, 1, 2]), {"task_goal": ["pick", "alternative"]}

            def step(self, vector):
                self.actions.append(vector)
                done = len(self.actions) == 5
                return frames([2 + len(self.actions)]), 0, False, done, {"status": "timeout"}

        class PolicyConnection:
            def call(self, func_name, obs=None):
                if func_name == "reset":
                    self.frames, self.inferences = [], 0
                elif func_name in ("update_obs", "update_obs_batch"):
                    self.frames.append(obs[0] if func_name.endswith("_batch") else obs)
                else:
                    self.inferences += 1
                    chunk = [{"arm_joint_state": np.arange(7), "ee_joint_state": [1]}] * 20
                    return [chunk] if func_name.endswith("_batch") else chunk

        for batch, deploy in ((False, eval_one_episode), (True, eval_one_episode_batch)):
            with self.subTest(batch=batch):
                sim, client = Simulator(), PolicyConnection()
                episode = Episode(sim, {"arm_dim": [7], "ee_dim": [1]}, None)
                deploy(episode, EpisodeRPC(client, episode, batch, horizon=2))
                self.assertEqual((episode.steps, client.inferences, episode.info["status"]), (5, 3, "timeout"))
                self.assertEqual([o["vision"]["cam_head"]["color"][0, 0, 0] for o in client.frames], list(range(7)))
                first = client.frames[0]
                self.assertEqual([o.get("demonstration_frame") for o in client.frames[:3]],
                                 [{"index": 0, "length": 2}, {"index": 1, "length": 2}, None])
                np.testing.assert_array_equal(first["vision"]["cam_wrist"]["color"][0, 0], [1, 5, 10])
                np.testing.assert_allclose(first["state"]["ee_joint_state"], [0.03])
                self.assertEqual(first["instruction"], "pick")
                for action in sim.actions:
                    np.testing.assert_array_equal(action, [0, 1, 2, 3, 4, 5, 6, 1])

    def test_reject_misaligned_demo_and_invalid_camera(self):
        raw = frames([0, 1])
        raw["front_rgb_list"].pop()
        with self.assertRaises(ValueError):
            observations(raw, "pick")
        raw = frames([0])
        raw["front_rgb_list"][0] = np.zeros((4, 5))
        with self.assertRaisesRegex(ValueError, "RGB"):
            observations(raw, "pick")
