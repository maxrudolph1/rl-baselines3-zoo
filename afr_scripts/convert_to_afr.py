import h5py
import torch
import numpy as np
import os

# Given a directory path, search for all .pth files and list their paths in "all_paths"
def find_pth_files(directory):
    all_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                all_paths.append(os.path.join(root, file))
    return all_paths

# Example usage:
# directory_path = "/your/directory/here"
# all_paths = find_pth_files(directory_path)

all_paths = find_pth_files('/u/mrudolph/documents/rl-baselines3-zoo/atari_data')

all_paths = ['/u/mrudolph/documents/rl-baselines3-zoo/atari_data/a2c_BreakoutNoFrameskip-v4_0_10000.pth']

train_split = 0.9

for atari_path in all_paths:
    print(f"Processing: {atari_path}")
    
    atari_data = torch.load(atari_path, weights_only=False)

    dones = np.concatenate([np.array([[False]]), atari_data['done'][:-1]])
    converted_atari_data = {
        "frames": atari_data['obs'].squeeze(),
        "action": atari_data['action'].squeeze(),
        "episode_index": np.cumsum(dones),
        "reward": atari_data['reward'].squeeze(),
        "background_id": np.zeros_like(atari_data['reward'].squeeze()),
        "state": np.zeros_like(atari_data['reward'].squeeze()),
        "step_type": np.zeros_like(atari_data['reward'].squeeze()),
        "done": atari_data['done'].squeeze(),
    }

    algo = atari_path.split('/')[-1].split('_')[0]
    game = atari_path.split('/')[-1].split('_')[1]
    size = atari_path.split('/')[-1].split('_')[-1].split('.')[0]

    afr_game_path = f'atari_data/{game}'
    os.makedirs(afr_game_path, exist_ok=True)

    N_episodes = converted_atari_data['episode_index'].max() + 1
    N_train = int(N_episodes * train_split)
    N_test = N_episodes - N_train
    n_samples = converted_atari_data['frames'].shape[0]
    
    train_indices = np.arange(n_samples)[converted_atari_data['episode_index'] < N_train]
    test_indices = np.arange(n_samples)[converted_atari_data['episode_index'] >= N_train]

    train_data = {k: v[train_indices] for k, v in converted_atari_data.items()}
    test_data = {k: v[test_indices] for k, v in converted_atari_data.items()}

    afr_train_path = f'{afr_game_path}/train_{algo}_{size}.hdf5'
    afr_test_path = f'{afr_game_path}/test_{algo}_{size}.hdf5'

    os.makedirs(os.path.dirname(afr_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(afr_test_path), exist_ok=True)

    with h5py.File(afr_train_path, 'w') as f:
        for key in train_data.keys():
            f[key] = train_data[key]
    with h5py.File(afr_test_path, 'w') as f:
        for key in test_data.keys():
            f[key] = test_data[key]
    
    print(f"Saved to: {afr_train_path} and {afr_test_path}")
