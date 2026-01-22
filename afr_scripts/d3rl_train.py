import h5py
import numpy as np
import d3rlpy
from huggingface_sb3 import EnvironmentName
from rl_zoo3.enjoy import create_test_env
from rl_zoo3.exp_manager import ExperimentManager
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, FrameStackObservation


def make_env(env_id):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env, screen_size=84, frame_skip=4, terminal_on_life_loss=False, grayscale_obs=True, noop_max=30)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    return env

data_path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/BreakoutNoFrameskip-v4_old/train_a2c_10000.hdf5'
data = h5py.File(data_path, 'r')

observations = np.moveaxis(data['frames'], -1, 1)
actions = data['action']
rewards = data['reward']
terminals = np.concatenate([np.diff(data['episode_index']), [1]])

env_name: EnvironmentName = "BreakoutNoFrameskip-v4"

log_dir = 'atari_data/runs'

# env = create_test_env(
#     "BreakoutNoFrameskip-v4",
#     n_envs=1,
#     stats_path=None,
#     seed=0,
#     log_dir=log_dir,
#     should_render=False,
#     hyperparams={},
#     env_kwargs={},
#     vec_env_cls=ExperimentManager.default_vec_env_cls,
# )

import pdb; pdb.set_trace()

dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    action_space=d3rlpy.constants.ActionSpace.DISCRETE,
    action_size=env.action_space.n,
)


# prepare algorithm
cql = d3rlpy.algos.DiscreteCQLConfig(
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    compile_graph=False,
).create(device='cuda:0')

# start training
cql.fit(
    dataset,
    n_steps=1000000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)},
)