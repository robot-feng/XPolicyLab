from pathlib import Path
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder

directory = Path('runs/robomme/native-parity')
directory.mkdir(parents=True, exist_ok=False)
for task in ['PickXtimes', 'VideoRepick']:
    env = None
    try:
        builder = BenchmarkEnvBuilder(env_id=task, dataset='test', action_space='joint_angle', gui_render=False, max_steps=2)
        env = builder.make_env_for_episode(0)
        raw, info = env.reset()
        np.savez_compressed(directory/f'{task}.npz', front=raw['front_rgb_list'][-1], wrist=raw['wrist_rgb_list'][-1],
                            joints=raw['joint_state_list'][-1], fingers=raw['gripper_state_list'][-1],
                            instruction=info['task_goal'][0])
        print(task, len(raw['front_rgb_list']), flush=True)
    finally:
        if env is not None:
            env.close()
