import argparse
import uuid
from datetime import datetime
from os.path import exists
from pathlib import Path

import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.atari_wrappers import WarpFrame
from tensorboard_callback import TensorboardCallback
from pygba import PyGBA
import tensorboard

from emerald_env import EmeraldEnv

import mgba.log
mgba.log.silence()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gba_path', type=str, default='roms/pokemon_emerald.gba')
    parser.add_argument('--init_state', type=str, default='saves/pokemon_emerald.new_game.sav')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--frameskip', type=int, default=24)
    parser.add_argument('--sticky_action_prob', type=float, default=0.2)
    parser.add_argument('--action_noise', type=float, default=0.0)
    parser.add_argument('--episode_length', type=int, default=2048 * 32)
    parser.add_argument('--early_stopping_patience', type=int, default=2048 * 4)
    parser.add_argument('--early_stopping_penalty', type=float, default=0.0)
    parser.add_argument('--learn_steps', type=int, default=40)
    parser.add_argument('--use_atari_wrapper', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--save_freq', type=int, default=2048 * 10)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--use_wandb_logging', action='store_true')
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

def make_gba_env(rank, env_conf, seed=0):
    def _init():
        gba = load_pokemon_emerald(env_conf['gba_path'], env_conf['init_state'])
        env = EmeraldEnv(
            gba,
            rank=rank,
            max_episode_steps=env_conf['max_steps'],
            frames_path=env_conf['frames_path'],
            frameskip=env_conf['frameskip'],
            sticky_action_probability=env_conf['sticky_action_probability'],
            action_noise=env_conf['action_noise'],
            early_stopping=env_conf['early_stopping_patience'] > 0,
            patience=env_conf['early_stopping_patience'],
            early_stopping_penalty=env_conf['early_stopping_penalty'],
        )
        # GBA screen is 240x160
        if env_conf['use_atari_wrapper'] == 1:
            env = WarpFrame(env, width=120, height=80)
        env.reset()
        return env
    set_random_seed(seed + rank)
    return _init


def main(args):
    ep_length = args.episode_length
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'{args.output_dir}/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_{sess_id}')
    sess_path.mkdir(parents=True, exist_ok=True)
    frames_path = sess_path / 'frames'

    env_config = dict(
        gba_path='roms/pokemon_emerald.gba',
        init_state=args.init_state,
        frames_path=frames_path,
        max_steps=ep_length, 
        frameskip=args.frameskip,
        sticky_action_probability=args.sticky_action_prob,
        action_noise=args.action_noise,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_penalty=args.early_stopping_penalty,
        use_atari_wrapper=args.use_atari_wrapper,
        reset_to_new_game_prob=0.5,
    )
    
    print(env_config)
    
    num_workers = args.num_workers  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_gba_env(i, env_config) for i in range(num_workers)])
    
    callbacks = [
        CheckpointCallback(save_freq=args.save_freq, save_path=sess_path, name_prefix='poke'),
        TensorboardCallback(),
    ]

    if args.use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    ppo_args = dict(
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=2048,
        n_epochs=3,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.1,
        target_kl=0.02,
        ent_coef=0.003,
        vf_coef=0.8,
    )
    
    if args.resume_checkpoint is not None:
        print(f'Resuming from checkpoint: {args.resume_checkpoint}')
        model = RecurrentPPO.load(args.resume_checkpoint, env=env, **ppo_args)
        model.n_envs = num_workers
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_workers
        model.rollout_buffer.reset()
    else:
        model = RecurrentPPO(
            'CnnLstmPolicy',
            env,
            verbose=1,
            tensorboard_log=sess_path,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                lstm_hidden_size=256,
                share_features_extractor=True,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(weight_decay=0.0, eps=1e-12),
            ),
            **ppo_args,
        )
    
    for i in range(args.learn_steps):
        model.learn(total_timesteps=(ep_length)*num_workers*1000, callback=CallbackList(callbacks))

    if args.use_wandb_logging:
        run.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)
