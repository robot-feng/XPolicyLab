from pathlib import Path

import numpy as np
import pytest

from xpolicy.core.config import load_service
from xpolicy.core.contracts import Observation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contracts():
    return load_service(ROOT / "configs/policies/mock.yaml")[1]


@pytest.fixture
def observation():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 210
    image[:, :, 1] = 50
    image[:, :, 2] = 15
    return Observation(
        images={"cam_head": image}, state=np.array([0.25, 0.02], dtype=np.float32),
        state_names=("arm_joint", "gripper"), state_units=("rad", "m"),
        instruction="hold", timestamp_s=1.5, clock_domain="test_clock",
        frame_id=7, episode_id="episode-a",
    )
