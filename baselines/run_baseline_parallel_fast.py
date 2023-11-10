from os.path import exists
from pathlib import Path
import uuid

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from pygba import PyGBA

from emerald_env import EmeraldEnv

import mgba.log
mgba.log.silence()

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
        env = EmeraldEnv(gba, frameskip=env_conf['frameskip'], max_episode_steps=env_conf['max_steps'])
        env.reset(seed=seed + rank)
        env.rank = rank
        return env
    set_random_seed(seed)
    return _init


def main():
    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
        # 'headless': True, 'save_final_state': True, 'early_stop': False,
        # 'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        # 'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
        # 'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
        # 'explore_weight': 3, # 2.5
        'gba_path': 'roms/pokemon_emerald.gba', 'init_state': 'roms/pokemon_emerald.sav',
        'max_steps': ep_length, 
        'frameskip': 23,
    }
    
    print(env_config)
    
    num_cpu = 24  # Also sets the number of episodes per training iteration
    # env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = SubprocVecEnv([make_gba_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    
    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
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

    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    file_name = 'session_e41c9eff/poke_38207488_steps' 
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()


if __name__ == '__main__':
    main()
