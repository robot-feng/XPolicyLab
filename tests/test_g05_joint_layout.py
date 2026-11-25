import unittest
from types import SimpleNamespace

import numpy as np

from XPolicyLab.policy.G05.model import Model
from XPolicyLab.utils.process_data import pack_robot_state


class G05JointLayout(unittest.TestCase):
    def test_single_and_dual_checkpoint_contracts(self):
        for sizes, sides, arm_key in [([7], ["right"], "arm"), ([6, 6], ["left", "right"], "control")]:
            with self.subTest(sides=sides):
                model = Model.__new__(Model)
                model.action_steps = 2
                model.action_type = "joint"
                model.embodiment_type = "test"
                dims = model.robot_action_dim_info = {"arm_dim": sizes, "ee_dim": [1] * len(sides)}
                spec = [
                    {"key": k, "raw_shape": n}
                    for side, size in zip(sides, sizes)
                    for k, n in [(f"{side}_{arm_key}", size), (f"{side}_gripper", 1)]
                ]
                cameras = [("front", "exterior", "cam_head")] + [
                    (f"wrist_{s}", f"wrist_{s}", "cam_wrist" if len(sides) == 1 else f"cam_{s}_wrist")
                    for s in sides
                ]
                model.processor = SimpleNamespace(
                    processors={
                        "test": SimpleNamespace(
                            shape_meta={
                                "state": spec,
                                "action": spec,
                                "images": [{"key": k, "camera_type": typ} for k, typ, _ in cameras],
                            }
                        )
                    }
                )
                x = np.arange(sum(sizes) + len(sides), dtype=np.float32)
                state = model._build_raw_state(x)
                actions = model._format_action_chunk({k: np.stack([v, v + 1]) for k, v in state.items()})
                for t, a in enumerate(actions):
                    np.testing.assert_array_equal(pack_robot_state({"state": a}, "joint", dims), x + t)
                obs = {
                    "vision": {
                        name: {"color": np.full((5, 6, 3), i, np.uint8)}
                        for i, (_, _, name) in enumerate(cameras)
                    }
                }
                images = model._build_raw_images(obs)
                for i, (k, _, _) in enumerate(cameras):
                    np.testing.assert_array_equal(images[k], np.full((3, 5, 6), i))
                if len(sizes) == 2:
                    model.robot_action_dim_info = {"arm_dim": [7], "ee_dim": [1]}
                    with self.assertRaises(ValueError):
                        model._build_raw_state(np.zeros(8))
