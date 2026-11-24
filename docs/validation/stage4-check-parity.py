import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from XPolicyLab.policy.Pi_05.robomme import load_policy
from XPolicyLab.policy.Pi_05.model import Model
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state
from XPolicyLab.utils.robomme_env_client import observations

root = Path.cwd()
checkpoint = root/'policy/Pi_05/checkpoints/robomme/79999'
directory = root/'runs/robomme/native-parity'
policy = load_policy(checkpoint, seed=0)
dimensions = get_robot_action_dim_info('panda')
report = {'scope': 'same real reset observation and RNG; native official inference vs XPolicyLab adapter; not task success',
          'seed': 0, 'samples': []}
for task in ['PickXtimes', 'VideoRepick']:
    data = np.load(directory/f'{task}.npz')
    native_obs = {'observation/state': np.concatenate([data['joints'], data['fingers'][:1]]).astype(np.float32),
                  'observation/image': data['front'], 'observation/wrist_image': data['wrist'],
                  'prompt': str(data['instruction'])}
    policy.reset()
    native = np.asarray(policy.infer(native_obs)['actions'])
    with patch.object(Model, 'get_model', return_value=policy):
        model = Model({'task_name': task, 'bench_name': 'RoboMME', 'env_cfg_type': 'panda', 'seed': 0})
    raw = {'front_rgb_list': [data['front']], 'wrist_rgb_list': [data['wrist']],
           'joint_state_list': [data['joints']], 'gripper_state_list': [data['fingers']]}
    model.reset()
    model.update_obs(next(observations(raw, str(data['instruction']))))
    adapted = np.stack([pack_robot_state({'state': a}, 'joint', dimensions) for a in model.get_action()])
    np.testing.assert_array_equal(adapted, native)
    np.savez(directory/f'{task}-actions.npz', native=native, adapted=adapted)
    report['samples'].append({'task': task, 'split': 'test', 'episode': 0, 'shape': list(native.shape),
                             'max_abs_diff': float(np.max(np.abs(native-adapted))), 'finite': bool(np.isfinite(native).all()),
                             'actions_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
                             'observation_sha256': hashlib.sha256((directory/f'{task}.npz').read_bytes()).hexdigest()})
    print(json.dumps(report['samples'][-1]), flush=True)
report['status'] = 'PASS'
(directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
