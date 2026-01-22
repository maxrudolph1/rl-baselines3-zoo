import h5py
import numpy as np
import imageio


path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/BreakoutNoFrameskip-v4/train_a2c_10000.hdf5'
output_path = 'afr_scripts/visualization.mp4'

with h5py.File(path, 'r') as f:
    frames = f['frames'][:]  # N x 84 x 84 x 4
    dones = f['done'][:]

# Extract first channel: N x 84 x 84
first_channel = frames[:, :, :, 0]

# Find frames that should flash red (5 frames after each done)
n_frames = len(first_channel)
flash_red = np.zeros(n_frames, dtype=bool)
done_indices = np.where(dones)[0]
for idx in done_indices:
    flash_red[idx:idx + 5] = True

# Create RGB frames
video_frames = []
for i in range(n_frames):
    gray_frame = first_channel[i]
    # Convert to RGB by stacking
    rgb_frame = np.stack([gray_frame, gray_frame, gray_frame], axis=-1)
    
    if flash_red[i]:
        # Flash red: set R channel high, reduce G and B
        rgb_frame = rgb_frame.astype(np.float32)
        rgb_frame[:, :, 0] = 255  # Full red
        rgb_frame[:, :, 1] = rgb_frame[:, :, 1] * 0.3  # Reduce green
        rgb_frame[:, :, 2] = rgb_frame[:, :, 2] * 0.3  # Reduce blue
        rgb_frame = rgb_frame.astype(np.uint8)
    
    video_frames.append(rgb_frame)

# Write MP4
imageio.mimwrite(output_path, video_frames, fps=30, codec='libx264')
print(f"Saved video to {output_path}")