import torch
import matplotlib.pyplot as plt
import numpy as np
path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/a2c_BreakoutNoFrameskip-v4_0_10000.pth'
data = torch.load(path, weights_only=False)
actions = data["action"]
obs = data["obs"]
first_obs = obs[3, 0]  # shape: (84,84,4)
plt.figure(figsize=(12, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(first_obs[:, :, i], cmap="gray")
    plt.title(f"Channel {i}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("first_obs_channels.png")
plt.close()


actions_np = actions.squeeze()  # Ensure 1D array
plt.figure(figsize=(6,4))
vals, counts = np.unique(actions_np, return_counts=True)
plt.bar(vals, counts / counts.sum())
plt.xlabel("Action")
plt.ylabel("Density")
plt.title("Density Histogram of Actions")
plt.xticks(vals)
plt.savefig("action_density_histogram.png")
plt.close()

