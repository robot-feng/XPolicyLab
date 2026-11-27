"""Paired replay on original MoveCube train episodes: the same grasp method in two layouts."""
import argparse
from contextlib import closing
import json
from pathlib import Path

import h5py
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder


def recorded_grasp(group):
    phases = {True: [], False: []}
    closure = {}
    for t in range(sum(k.startswith("timestep_") for k in group)):
        step = group[f"timestep_{t}"]
        demo = bool(step["info/is_video_demo"][()])
        label = step["info/simple_subgoal"].asstr()[()]
        if label not in ("Pick up the cube", "place the cube onto the target"):
            continue
        phases[demo].append(step["action/joint_action"][()])
        if demo not in closure and label == "Pick up the cube" and step["obs/is_gripper_close"][()]:
            closure[demo] = step["obs/eef_state"][()][:3].tolist()
    if set(closure) != set(phases) or not all(phases.values()):
        raise ValueError("Expected a recorded pick-and-place phase in both layouts")
    return phases, closure


def run(source, count):
    builder = BenchmarkEnvBuilder(env_id="MoveCube", dataset="train", action_space="joint_angle",
                                  gui_render=False, max_steps=1300)
    rows = []
    with h5py.File(source) as h5:
        for eid in range(builder.get_episode_num()):
            group = h5[f"episode_{eid}"]
            if group["timestep_0/info/simple_subgoal"].asstr()[()] != "Pick up the cube":
                continue
            seed = builder.resolve_episode(eid)[0]
            assert int(group["setup/seed"][()]) == seed
            phases, closure = recorded_grasp(group)
            row = dict(episode=eid, seed=seed, demonstration_close_xyz_m=closure[True],
                       execution_close_xyz_m=closure[False])
            row["close_xy_distance_cm"] = float(100 * np.linalg.norm(
                np.subtract(closure[True][:2], closure[False][:2])))
            initial = None
            for demo, actions in phases.items():
                with closing(builder.make_env_for_episode(eid)) as env:
                    obs, _ = env.reset()
                    q0 = obs["joint_state_list"][-1]
                    if initial is not None:
                        np.testing.assert_allclose(initial, q0, atol=1e-5, rtol=0)
                    initial = np.array(q0, copy=True)
                    # Commands include the recorded gripper. Hold the last command until native termination.
                    for t in range(1302):
                        _, _, term, trunc, info = env.step(actions[min(t, len(actions)-1)].copy())
                        if term or trunc:
                            break
                    else:
                        raise RuntimeError("Missing native termination at the configured budget")
                    row["replay_demo" if demo else "replay_execution_oracle"] = {
                        "status": info["status"], "control_steps": t+1, "recorded_commands": len(actions)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if len(rows) == count:
                return rows
    raise ValueError("Insufficient grasp episodes in the training split")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be positive")
    args.output.write_text(json.dumps(run(args.h5, args.count), indent=2) + "\n")
