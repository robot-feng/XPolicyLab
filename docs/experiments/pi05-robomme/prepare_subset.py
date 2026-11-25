"""Build execution-only samples for RoboMME's history-free pi05_baseline loader."""
import argparse
import json
from pathlib import Path
import pickle

import h5py
import numpy as np

from mme_vla_suite.dataset_builder.build_robomme_dataset import get_action_chunk
from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('h5', type=Path)
    parser.add_argument('train_metadata', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--episodes', type=int, default=4)
    args = parser.parse_args()
    records = json.loads(args.train_metadata.read_text())['records']
    if not 1 <= args.episodes <= len(records) or args.train_metadata.parent.name != 'train':
        parser.error('Use train split metadata and a positive number of available episodes')
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'data'
    meta = args.output / 'meta'
    data.mkdir()
    meta.mkdir()
    count = 0
    manifest = []
    with h5py.File(args.h5, 'r') as h5:
        for record in records[:args.episodes]:
            episode = h5[f"episode_{record['episode']}"]
            seed = int(episode['setup/seed'][()])
            if seed != record['seed'] or args.h5.stem != f"record_dataset_{record['task']}":
                raise ValueError(f'Wrong dataset or split for {record}')
            prompt = episode['setup/task_goal'].asstr()[0].lower()
            start = first_execution_step(episode)
            end = sum(key.startswith('timestep_') for key in episode)
            for step in range(start, end):
                ts = episode[f'timestep_{step}']
                if ts['info/is_video_demo'][()]:
                    raise ValueError(f'Unexpected demonstration frame at {step}')
                obs = ts['obs']
                sample = dict(image=obs['front_rgb'][()], wrist_image=obs['wrist_rgb'][()],
                              state=np.concatenate([obs['joint_state'][()], obs['gripper_state'][:1]]),
                              actions=get_action_chunk(episode, step), prompt=prompt)
                # The upstream loader drops these two fields in the history-free configuration.
                for key in ('simple_subgoal_online', 'grounded_subgoal_online'):
                    sample[key] = ts['info'][key].asstr()[()]
                with (data / f'{count}.pkl').open('xb') as stream:
                    pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
                count += 1
            manifest.append(dict(episode=record['episode'], seed=seed, first_execution=start, samples=end-start))
    (meta / 'stats.json').write_text(json.dumps(dict(execution_samples=count, episodes=manifest), indent=2) + '\n')
    print((meta / 'stats.json').read_text())


if __name__ == '__main__':
    main()
