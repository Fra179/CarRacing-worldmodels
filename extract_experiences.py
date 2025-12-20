import gymnasium as gym
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import torch
from torchvision import transforms


def preprocess_frame(frame, img_size):
    """
    Converts 96x96 observation to 64x64 using torchvision.
    CarRacing-v2 gives (96, 96, 3).
    """
    # Convert numpy array (H, W, C) to tensor and permute to (C, H, W)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

    # Resize
    resizer = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)
    resized = resizer(frame_tensor)

    # Convert back to numpy (H, W, C) and uint8
    return resized.permute(1, 2, 0).numpy().astype(np.uint8)


def generate_data(dir_name, num_episodes, img_size):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")

    env = gym.make("CarRacing-v2", render_mode=None)  # Set render_mode="human" to watch

    start_time = time.time()
    idx = 0

    for i in tqdm(range(num_episodes)):
        obs, _ = env.reset()

        # Storage for this episode
        rollout_obs = []
        rollout_action = []

        done = False
        truncated = False

        for _ in range(50):
            # skip the first 50 frames
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs

        while not (done or truncated):
            # Paper uses a random policy
            action = env.action_space.sample()

            # Step the environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Process and store
            # Note: We store the observation *before* the action took place
            # or the one resulting from it. Usually standard is (obs, action, next_obs)
            # but for VAE we just need a massive pile of frames.
            processed_obs = preprocess_frame(obs, img_size)

            filename = os.path.join(dir_name, f"obs_{idx}.npy")
            np.save(filename, processed_obs)
            idx += 1

            rollout_obs.append(processed_obs)
            rollout_action.append(action)

            obs = next_obs

        # Convert to numpy arrays
        # Shapes: (T, 64, 64, 3) and (T, 3)
        rollout_obs = np.array(rollout_obs, dtype=np.uint8)
        rollout_action = np.array(rollout_action, dtype=np.float16)  # Save space

        # Save compressed to save disk space
        filename = os.path.join(dir_name, f"rollout_{i + 1}.npz")
        np.savez_compressed(filename, obs=rollout_obs, action=rollout_action)

        if (i + 1) % 10 == 0:
            print(f"Saved Episode {i + 1}/{num_episodes} to {filename}")

    env.close()
    print(f"Finished. Total time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Generate CarRacing-v2 experience data')
    parser.add_argument('--dir', type=str, default='car_racing_data', help='Directory to save data')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to generate')
    parser.add_argument('--img_size', type=int, default=64, help='Size to resize images to (img_size x img_size)')
    args = parser.parse_args()

    generate_data(args.dir, args.episodes, args.img_size)

if __name__ == "__main__":
    main()