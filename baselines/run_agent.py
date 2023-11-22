import argparse
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from pygba import PyGBA
import torch
import tqdm

from emerald_env import EmeraldEnv
from render_episode_trajectory import render_video, plot_rewards

import mgba.log
mgba.log.silence()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='outputs/2023-11-21/18-04-51_b472560c/poke_8847360_steps.zip')
    parser.add_argument('--output_dir', type=str, default='outputs/inference')
    parser.add_argument('--resolution', type=str, default='480:320')
    parser.add_argument('--speedup', type=float, default=16.0)
    parser.add_argument('--episode_length', type=int, default=1024)
    parser.add_argument('--gba_path', type=str, default='roms/pokemon_emerald.gba')
    parser.add_argument('--init_state', type=str, default='saves/pokemon_emerald.new_game.sav')
    parser.add_argument('--frameskip', type=int, default=24)
    parser.add_argument('--sticky_action_prob', type=float, default=0.2)
    parser.add_argument('--action_noise', type=float, default=0.0)
    parser.add_argument('--early_stopping_patience', type=int, default=2048 * 4)
    parser.add_argument('--early_stopping_penalty', type=float, default=0.0)
    parser.add_argument('--use_atari_wrapper', type=int, default=1)
    parser.add_argument('--deterministic_actions', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_pokemon_emerald(gba_file: str, save_file: str | None):
    gba = PyGBA.load(gba_file, save_file=save_file)
    if save_file is not None:
        # skip loading screen
        for _ in range(16):
            gba.press_a(30)
        gba.wait(60)
    else:
        # skip loading screen and character creation
        gba.wait(600)
        for _ in range(120):
            gba.press_a(30)
        gba.wait(720)
    return gba

def make_gba_env(env_conf, frames_path: str | None = None):
    gba = load_pokemon_emerald(env_conf['gba_path'], env_conf['init_state'])
    env = EmeraldEnv(
        gba,
        max_episode_steps=env_conf['max_steps'],
        frameskip=env_conf['frameskip'],
        save_episode_frames=(frames_path is not None),
        frames_path=frames_path,
        frame_save_freq=1,
        sticky_action_probability=env_conf['sticky_action_probability'],
        action_noise=env_conf['action_noise'],
        early_stopping=env_conf['early_stopping_patience'] > 0,
        patience=env_conf['early_stopping_patience'],
        early_stopping_penalty=env_conf['early_stopping_penalty'],
        save_episode_trajectory=True,
        episode_trajectory_path=env_conf['session_path'] / "trajectories",
        verbose=False,
    )
    # GBA screen is 240x160
    if env_conf['use_atari_wrapper'] == 1:
        env = WarpFrame(env, width=120, height=80)
    return env


def main(args):
    ep_length = args.episode_length
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'{args.output_dir}/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_{sess_id}')
    sess_path.mkdir(parents=True, exist_ok=True)

    with (sess_path / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading checkpoint: {args.checkpoint_path}')
    model = RecurrentPPO.load(args.checkpoint_path)
    model.policy.set_training_mode(False)
    model.policy.to(device)

    env_config = dict(
        gba_path='roms/pokemon_emerald.gba',
        init_state=args.init_state,
        session_path=sess_path,
        max_steps=ep_length, 
        frameskip=args.frameskip,
        sticky_action_probability=args.sticky_action_prob,
        action_noise=args.action_noise,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_penalty=args.early_stopping_penalty,
        use_atari_wrapper=args.use_atari_wrapper,
    )
    print(env_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_path = Path(tmpdir) / "frames"
        frames_path.mkdir()
        env = make_gba_env(env_config, frames_path=frames_path)
        obs, info = env.reset(seed=args.seed)

        state = None
        rewards = []
        total_reward = 0
        with tqdm.tqdm(total=args.episode_length) as pbar:
            for _ in range(args.episode_length):
                action, state = model.predict(obs, state, deterministic=(args.deterministic_actions == 1))
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                total_reward += reward
                pbar.update(1)
                pbar.set_postfix_str(f"reward={total_reward:.2f}")
                if done:
                    pbar.close()
                    break

        render_video(
            frames_path,
            sess_path / "video.mp4",
            resolution=args.resolution,
            frameskip=args.frameskip,
            speedup=args.speedup,
        )
        plot_rewards(rewards, sess_path / "rewards.png")

        print(f"Saved video to {sess_path / 'video.mp4'}")
        print(f"Saved rewards plot to {sess_path / 'rewards.png'}")

        


if __name__ == '__main__':
    args = parse_args()
    main(args)
