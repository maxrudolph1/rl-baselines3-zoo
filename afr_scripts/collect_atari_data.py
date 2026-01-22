import gymnasium as gym
import ale_py
from rl_zoo3.utils import StoreDict, get_model_path
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
import numpy as np
import time
from collections import defaultdict
import torch as th
from tqdm import tqdm

gym.register_envs(ale_py)

from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, FrameStackObservation


def make_env(env_id):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env, screen_size=84, frame_skip=4, terminal_on_life_loss=False, grayscale_obs=True, noop_max=30)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    return env

algo = 'a2c'
folder = 'rl-trained-agents/'
env_name = 'BreakoutNoFrameskip-v4'
exp_id = 0
load_best = False
load_checkpoint = None
load_last_checkpoint = False
time_steps = 1000
no_render = True

_, model_path, log_path = get_model_path(
    exp_id,
    folder,
    algo,
    env_name,
    load_best,
    load_checkpoint,
    load_last_checkpoint,
)

env = make_env(env_name)

model = ALGOS[algo].load(model_path, custom_objects={})


obs, _ = env.reset()

# Deterministic by default except for atari games
stochastic = True
deterministic = not stochastic

episode_reward = 0.0
episode_rewards, episode_lengths = [], []
ep_len = 0
# For HER, monitor success rate
successes = []
lstm_states = None
episode_start = np.ones((1,), dtype=bool)

generator = range(time_steps)

generator = tqdm(generator)

try:
    trajectory_data = defaultdict(list) # obs, action, reward, done, info
    start_time = time.time()
    for _ in generator:
                    
    
        action, lstm_states = model.predict(
            obs,  # type: ignore[arg-type]
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        trajectory_data["obs"].append(obs)
        trajectory_data["action"].append(action)

        
        obs, reward, done, _, infos = env.step(action)
        trajectory_data["reward"].append(reward)
        trajectory_data["done"].append(done)
        trajectory_data["info"].append(infos)
        
        episode_start = done

        if not no_render:
            env.render("human")

        episode_reward += reward
        ep_len += 1

                    
    end_time = time.time()
    print('episode reward: ', episode_reward)
    print('episode length: ', ep_len)

except KeyboardInterrupt:
    pass

    
trajectory_data = {k: np.array(v) for k, v in trajectory_data.items()}
th.save(trajectory_data, f"atari_data/{algo}_{env_name}_{exp_id}_{time_steps}.pth")

import imageio

# Get the first 300 observations (frames)
frames = trajectory_data["obs"][:300]  # shape: (n, 4, 84, 84) - stacked grayscale frames
frames_arr = np.array(frames)

# frames_arr shape is (300, 4, 84, 84) - use the last frame from each stack
imgs = []
for i in range(min(300, frames_arr.shape[0])):
    frame = frames_arr[i, -1]  # Get the most recent frame from the stack (84, 84)
    # Normalize if not uint8
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    # Convert grayscale to RGB by repeating the channel
    rgb = np.stack([frame, frame, frame], axis=-1)
    imgs.append(rgb)

imageio.mimsave("atari_movie.mp4", imgs, fps=30)

    
env.close()
