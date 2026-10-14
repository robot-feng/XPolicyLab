from pathlib import Path

import numpy as np

from tools.validate_g2_model_boundary import arrays, bundle
from xpolicy.diagnostics.golden import compare_bundles, load_bundle, save_bundle


def test_tool_preserves_separate_state_action_axes(tmp_path):
    reference = save_bundle(tmp_path / "training", scope="synthetic_test", sample={"frame_id": 2}, provenance={}, layers=[
        {"name": "O1", "metadata": {"state_names": ["state-a", "state-b"], "action_names": ["action-a", "action-b"]},
         "arrays": {"state": np.ones((1, 2))}},
        {"name": "O2", "metadata": {"boundary": "packed"},
         "arrays": {"state": np.ones((1, 2)), "action_labels": np.ones((3, 2))},
         "axes": {"state": {"1": ["state-a", "state-b"]}, "action_labels": {"1": ["action-a", "action-b"]}}},
    ])
    manifest, values = load_bundle(reference)
    current = bundle(tmp_path / "current", manifest, arrays(manifest, values, "O2"), manifest.layers[1].metadata)
    assert compare_bundles(reference, current, layers=["O2"])["status"] == "PASS"
    prediction = bundle(tmp_path / "prediction", manifest, {"env_actions": np.ones((1, 3, 2))}, {}, name="A0")
    desc = load_bundle(prediction)[0].layers[0].arrays["env_actions"]
    assert desc.axes["2"] == ("action-a", "action-b")
