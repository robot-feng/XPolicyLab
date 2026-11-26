"""Executed history must describe the corrected commands, including a partial receipt."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from g05.data.native_episodes import hold_pad
from XPolicyLab.policy.KnowAct.demo_head import DemoHead
from XPolicyLab.policy.KnowAct.demonstration import state_groups


class TestDemoConditioning(unittest.TestCase):
    def test_history_keeps_dispatched_prefix(self):
        @dataclass
        class State:
            pending_action: dict

        normalizer = SimpleNamespace(forward=lambda data: {"action": {k: v * 2 for k, v in data["action"].items()}})
        merger = SimpleNamespace(forward=lambda data: {"action": torch.cat(list(data["action"].values()), dim=-1)})
        processor = SimpleNamespace(shape_meta={"action": [{"key": k} for k in ("right_arm", "right_gripper")]},
                                    normalizer=normalizer, action_state_merger=merger)
        base = {"right_arm": np.zeros((3, 2)), "right_gripper": np.array([[1.], [0.], [-1.]])}
        state = State({"action": {"value": torch.zeros(3, 3)}})
        head = DemoHead.__new__(DemoHead)
        head.weights = np.array([0., 1., 0., 0., 0.])
        demo = np.arange(10, dtype=np.float32).reshape(5, 2)
        action, candidate, _ = head.apply(demo, demo[0], base, state, processor, 0, 0)
        self.assertFalse(state.pending_action["action"]["value"].any())
        sent = np.column_stack([action["right_arm"], action["right_gripper"]])
        np.testing.assert_array_equal(hold_pad(candidate.pending_action["action"]["value"], 2).numpy(),
                                      2 * sent[[0, 1, 1]])

    def test_joint_groups_preserve_visual_anchors_and_order(self):
        grouped = state_groups(48, 4, 16)
        self.assertEqual(list(grouped), [0, 15, 31, 47])
        samples = [t for group in grouped.values() for t in group]
        self.assertEqual(len(samples), 16)
        self.assertEqual(samples, sorted(set(samples)))
        self.assertTrue(all(times[-1] == end for end, times in grouped.items()))
        self.assertEqual(state_groups(2, 4, 16), {0: [0], 1: [1]})
