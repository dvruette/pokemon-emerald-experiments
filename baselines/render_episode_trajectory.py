import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygba import PyGBA
from mgba._pylib import ffi
import tqdm

from emerald_env import EmeraldEnv


import mgba.log
mgba.log.silence()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_path', type=str, default='outputs/2023-11-22/17-09-28_32539bdd/trajectories/00/episode_0000')
    parser.add_argument('--output_path', type=str, default='outputs/2023-11-22/17-09-28_32539bdd/trajectories/00/episode_0000')
    parser.add_argument('--gba_path', type=str, default='roms/pokemon_emerald.gba')
    parser.add_argument('--speedup', type=float, default=16.0)
    parser.add_argument('--resolution', type=str, default='480:320')
    return parser.parse_args()

def make_gba_env(
    gba_file: str,
    frames_path: str,
    trajectory_path: Path,
    max_steps: int = 32 * 2048,
    frameskip: int = 24,
    repeat_action_probability: float = 0.0,
    action_noise: float = 0.0,
    seed: int = 0,
):
    gba = PyGBA.load(gba_file)
    env = EmeraldEnv(
        gba,
        max_episode_steps=max_steps,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        action_noise=action_noise,
        save_episode_frames=True,
        frame_save_freq=1,
        frames_path=frames_path,
        save_episode_trajectory=True,
        episode_trajectory_path=trajectory_path,
        verbose=False,
    )
    env.reset(seed)
    return env

def load_trajectory(gba_path: str, input_traj_path: Path, output_traj_path: Path, frames_path: Path):
    with (input_traj_path / "config.json").open("r") as f:
        config = json.load(f)

    with (input_traj_path / "initial_state").open("rb") as f:
        initial_state_bytes = f.read()
        initial_state = ffi.new(f"unsigned char[{len(initial_state_bytes)}]", initial_state_bytes)

    env = make_gba_env(
        gba_path,
        frames_path=frames_path,
        trajectory_path=output_traj_path,
        max_steps=config["max_steps"],
        frameskip=config["frameskip"],
        repeat_action_probability=config.get("repeat_action_probability", 0),
        action_noise=config.get("action_noise", 0),
        seed=config.get("seed", 0),
    )

    env.gba.core.reset()
    env.gba.core.load_raw_state(initial_state)

    with (input_traj_path / "actions.txt").open("r") as f:
        lines = f.readlines()
    actions = [int(l.strip()) for l in lines]

    return config, env, actions

def simulate_trajectory(env: EmeraldEnv, actions: list[int]):
    rewards = []
    for action in tqdm.tqdm(actions):
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
    print()
    return rewards

def render_video(
    frames_path: Path,
    output_path: Path,
    resolution: str = "240:160",
    base_framerate: int = 60,
    frameskip: int = 24,
    speedup: float = 1.0,
):
    framerate = round(base_framerate / frameskip * speedup)
    ffmpeg_command = " ".join([
        "ffmpeg -y",
        f"-framerate {framerate}",
        f"-i {str(frames_path)}/00/%06d.png",
        f"-vf scale={resolution}",
        "-r 30",
        "-c:v libx264",
        "-pix_fmt yuv420p",
        "-sws_flags neighbor",
        f"{str(output_path)}",
    ])

    print("Running:", ffmpeg_command)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.system(ffmpeg_command)

def plot_rewards(rewards: list[float], output_path: Path):
    sns.set_theme()
    cum_rewards = np.cumsum(rewards)
    plt.plot(cum_rewards)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

def main(args):
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_path = Path(tmpdir) / "frames"
        frames_path.mkdir(parents=True, exist_ok=True)

        output_path = Path(args.output_path)

        config, env, actions = load_trajectory(args.gba_path, Path(args.trajectory_path), output_path / "trajectories", frames_path)
        # actions = actions[1:] # skip first action due to bug in EmeraldEnv
        rewards = simulate_trajectory(env, actions)
        render_video(
            frames_path,
            Path(args.output_path) / "video.mp4",
            resolution=args.resolution,
            frameskip=config["frameskip"],
            speedup=args.speedup,
        )
        plot_rewards(rewards, Path(args.output_path) / "rewards.png")

        print(f"Video saved to: {args.output_path}/video.mp4")

if __name__ == '__main__':
    main(parse_args())
