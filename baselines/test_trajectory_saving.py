import json
import uuid
from datetime import datetime
from pathlib import Path
from pygba import PyGBA

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from mgba._pylib import ffi
import tqdm
from emerald_env import EmeraldEnv
from train import load_pokemon_emerald
from render_episode_trajectory import load_trajectory

import mgba.log
mgba.log.silence()


def make_gba_env(rank, trajectory_path, episode_len, seed=0):
    def _init():
        gba = load_pokemon_emerald(
            'roms/pokemon_emerald.gba',
            'saves/pokemon_emerald.new_game.sav',
        )
        env = EmeraldEnv(
            gba,
            rank=rank,
            max_episode_steps=episode_len,
            frameskip=24,
            repeat_action_probability=0.2,
            action_noise=0.1,
            save_episode_trajectory=True,
            episode_trajectory_path=trajectory_path,
            verbose=False,
        )
        # GBA screen is 240x160
        env = WarpFrame(env, width=120, height=80)
        env.reset(seed=(seed + rank))
        return env
    return _init


def generate_trajectory(sess_path: Path, num_episodes: int, episode_len: int):

    traj_path = sess_path / "trajectories"
    
    num_workers = 1  # Also sets the number of episodes per training iteration
    env = DummyVecEnv([make_gba_env(i, traj_path, episode_len) for i in range(num_workers)])
    
    model = RecurrentPPO('CnnLstmPolicy', env, n_steps=episode_len)
    model.learn(total_timesteps=num_episodes * episode_len)

    return traj_path


def load_trajectory(sess_path: Path, gba_path: str, trajectory_path: Path):
    with (trajectory_path / "config.json").open("r") as f:
        config = json.load(f)

    with (trajectory_path / "initial_state").open("rb") as f:
        initial_state_bytes = f.read()
        initial_state = ffi.new(f"unsigned char[{len(initial_state_bytes)}]", initial_state_bytes)


    gba = PyGBA.load(gba_path)
    env = EmeraldEnv(
        gba,
        max_episode_steps=config["max_steps"],
        frameskip=config["frameskip"],
        repeat_action_probability=config.get("repeat_action_probability", 0),
        action_noise=config.get("action_noise", 0),
        save_episode_trajectory=True,
        episode_trajectory_path=(sess_path / "trajectories_rec"),
        verbose=False,
    )
    env.reset(config.get("seed", 0))

    assert bytes(initial_state) == initial_state_bytes

    env.gba.core.reset()
    env.gba.core.load_raw_state(initial_state)

    assert bytes(initial_state) == bytes(env.gba.core.save_raw_state())


    with (trajectory_path / "actions.txt").open("r") as f:
        lines = f.readlines()
    actions = [int(l.strip()) for l in lines]

    with (trajectory_path / "log.txt").open("r") as f:
        logs = f.readlines()

    with (trajectory_path / "final_state").open("rb") as f:
        final_state = f.read()

    return config, env, actions, logs, initial_state_bytes, final_state


def main():
    num_episodes = 4
    episode_len = 32

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'outputs/test/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_{sess_id}')
    sess_path.mkdir(parents=True, exist_ok=True)

    trajectory_path = generate_trajectory(sess_path, num_episodes, episode_len)

    for episode in tqdm.trange(num_episodes):
        traj_path = trajectory_path / f"episode_{episode:04d}"
        config, env_rec, actions, logs, initial_state, final_state = load_trajectory(sess_path, 'roms/pokemon_emerald.gba', traj_path)

        before_state = env_rec.gba.core.save_raw_state()
        # env_rec.gba.core.load_raw_state(before_state)
        # assert bytes(before_state) == bytes(env_rec.gba.core.save_raw_state())

        # assert initial_state == bytes(before_state)
        # env_rec.gba.core.load_raw_state(before_state)

        rewards = []
        for action in tqdm.tqdm(actions):
            obs, reward, done, trunc, info = env_rec.step(action)
            rewards.append(reward)

        assert done

        rec_state_1 = bytes(env_rec.gba.core.save_raw_state())


        with open(sess_path / "trajectories_rec" / f"episode_{episode:04d}" / "actions.txt") as f:
            actions_rec = [int(l.strip()) for l in f.readlines()]

        assert all(a == b for a, b in zip(actions, actions_rec)), ", ".join([str((i, a, b)) for i, (a, b) in enumerate(zip(actions, actions_rec)) if a != b])
        
        assert rec_state_1 == final_state

        env_rec.save_episode_trajectory = False
        env_rec.reset(seed=config["seed"])
        env_rec.gba.core.reset()
        env_rec.gba.core.load_raw_state(before_state)

        rewards = []
        for action in tqdm.tqdm(actions):
            obs, reward, done, trunc, info = env_rec.step(action)
            rewards.append(reward)

        rec_state_2 = bytes(env_rec.gba.core.save_raw_state())

        assert rec_state_1 == rec_state_2


if __name__ == '__main__':
    main()
