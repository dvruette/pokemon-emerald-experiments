import json
import re
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from pygba import PyGBAEnv, PyGBA
from pygba.utils import KEY_MAP

from emerald_wrapper import CustomEmeraldWrapper


def clip_reward(reward: float, pos_scale: float = 5.0, neg_scale: float = 0.3):
    # from IMPALA: https://arxiv.org/abs/1802.01561
    reward = np.tanh(reward)
    if reward > 0:
        reward *= pos_scale
    else:
        reward *= neg_scale
    return reward

class EmeraldEnv(PyGBAEnv):
    def __init__(
        self,
        gba: PyGBA,
        rank: int = 0,
        save_episode_trajectory: bool = False,
        episode_trajectory_path: str | Path | None = None,
        save_episode_frames: bool = False,
        frames_path: str | Path | None = None,
        frame_save_freq: int = 1,
        early_stopping: bool = False,
        patience: int = 1024,
        early_stopping_penalty: float = 0.0,
        action_noise: float = 0.0,
        reset_to_new_game_prob: float = 1.0,
        save_intermediate_state_prob: float = 1e-3,
        reward_clipping: bool = False,
        reward_scale: float = 1.0,
        verbose: bool = True,
        wrapper_kwargs: dict = {},
        **kwargs,
    ):
        if save_episode_trajectory and episode_trajectory_path is None:
            raise ValueError("episode_trajectory_path must be specified if save_episode_trajectory is True")

        self.rank = rank
        self.save_episode_trajectory = save_episode_trajectory
        self.episode_trajectory_path = episode_trajectory_path
        self.save_episode_frames = save_episode_frames
        self.frames_path = frames_path
        self.frame_save_freq = frame_save_freq
        self.early_stopping = early_stopping
        self.patience = patience
        self.early_stopping_penalty = early_stopping_penalty
        self.action_noise = action_noise
        self.reset_to_new_game_prob = reset_to_new_game_prob
        self.save_intermediate_state_prob = save_intermediate_state_prob
        self.reward_clipping = reward_clipping
        self.reward_scale = reward_scale
        self.verbose = verbose

        self._intermediate_state = None
        self._curr_trajectory_path = None
        self._curr_seed = None
        self._total_reward = 0
        self._max_reward = 0
        self._max_reward_step = 0
        self._rng = np.random.default_rng()

        game_wrapper = CustomEmeraldWrapper(**wrapper_kwargs)
        super().__init__(gba, game_wrapper, **kwargs)

        self.arrow_keys = [None, "up", "down", "right", "left"]
        # self.buttons = [None, "A", "B", "select", "start", "L", "R"]
        self.buttons = [None, "A", "B"]

        # cartesian product of arrows and buttons, i.e. can press 1 arrow and 1 button at the same time
        self.actions = [(a, b) for a in self.arrow_keys for b in self.buttons]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        
        self.agent_stats = []

    def step(self, action_id):
        info = {}
        
        if self.save_episode_trajectory and self._curr_trajectory_path is not None:
            if self._step == 0:
                self._curr_trajectory_path.mkdir(parents=True, exist_ok=True)
                initial_state = self.gba.core.save_raw_state()
                with open(self._curr_trajectory_path / "initial_state", "wb") as f:
                    f.write(bytes(initial_state))

                with open(self._curr_trajectory_path / "config.json", "w") as f:
                    json.dump({
                        "max_steps": self.max_episode_steps,
                        "frameskip": self.frameskip,
                        "repeat_action_probability": self.repeat_action_probability,
                        "action_noise": self.action_noise,
                        "seed": self._curr_seed
                    }, f, indent=4)

        if self._rng.random() < self.action_noise:
            action_id = self._rng.integers(len(self.actions))
        actions = self.get_action_by_id(action_id)
        actions = [KEY_MAP[a] for a in actions if a is not None]

        if self.frames_path is not None and self.frame_save_freq > 0 and (self._step + 1) % self.frame_save_freq == 0:
            img = self._framebuffer.to_pil().convert("RGB")
            if self.save_episode_frames:
                out_path = Path(self.frames_path) / f"{self.rank:02d}" / f"{self._step:06d}.png"
                if self._step == 0 or self._step + 1 == self.frame_save_freq:
                    # delete old frames
                    if out_path.parent.exists():
                        shutil.rmtree(out_path.parent)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)
            # always save the most recent frame
            thumbnail_path = Path(self.frames_path) / f"{self.rank:02d}.png"
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(thumbnail_path)

        if self._rng.random() > self.repeat_action_probability:
            self.gba.core.set_keys(*actions)

        if isinstance(self.frameskip, tuple):
            frameskip = self._rng.integers(*self.frameskip)
        else:
            frameskip = self.frameskip

        for _ in range(frameskip + 1):
            self.gba.core.run_frame()
            pass
        observation = self._get_observation()

        # save intermediate state
        if self._rng.random() < self.save_intermediate_state_prob:
            self._intermediate_state = self.gba.core.save_raw_state()

        # reward calculation
        reward = 0
        if self.game_wrapper is not None:
            reward = self.game_wrapper.reward(self.gba, observation)
            info.update(self.game_wrapper.info(self.gba, observation))
        if self.early_stopping and self._step - self._max_reward_step > self.patience:
            reward -= self.early_stopping_penalty

        reward = reward * self.reward_scale
        if self.reward_clipping:
            reward = clip_reward(reward)

        self._total_reward += reward
        if self._total_reward > self._max_reward:
            self._max_reward = self._total_reward
            self._max_reward_step = self._step

        # the tensorboard will read out the agent_stats list and plot it
        self.agent_stats.append(info["rewards"])

        done = self.check_if_done()
        truncated = self.check_if_truncated()

        reward_display = " | ".join(f"{re.sub(r'_rew(ard)?', '', k)}={v:4.1f}" for k, v in info["rewards"].items())
        reward_display = f"step={self._step:5d} | {reward_display}"

        if self.save_episode_trajectory and self._curr_trajectory_path is not None:
            with open(self._curr_trajectory_path / "actions.txt", "a") as f:
                f.write(str(action_id) + "\n")

            with open(self._curr_trajectory_path / "log.txt", "a") as f:
                f.write(reward_display + "\n")

            if done:
                final_state = self.gba.core.save_raw_state()
                with open(self._curr_trajectory_path / "final_state", "wb") as f:
                    f.write(bytes(final_state))

        if self.verbose:
            print("\r " + reward_display, end="", flush=True)

        self._step += 1
        return observation, reward, done, truncated, info
    
    def check_if_truncated(self):
        if self.max_episode_steps is not None and self._step + 1 >= self.max_episode_steps:
            return True
        if self.early_stopping and self._step - self._max_reward_step > self.patience:
            return True
        return False
    
    def check_if_done(self):
        if self.check_if_truncated():
            return True
        if self.game_wrapper is not None:
            observation = self._get_observation()
            return self.game_wrapper.game_over(self.gba, observation)
        return False
    
    def get_last_agent_stats(self):
        if len(self.agent_stats) == 0:
            return None
        return self.agent_stats[-1]

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(2 ** 32 - 1)

        super().reset(seed=seed)
        set_random_seed(seed)
        self._rng = np.random.default_rng(seed)
        self._curr_seed = seed

        self._total_reward = 0
        self._max_reward = 0
        self._max_reward_step = 0
        self.agent_stats = []

        if self._intermediate_state is not None and np.random.random() >= self.reset_to_new_game_prob:
            self.gba.core.load_raw_state(self._intermediate_state)
            # self.gba.core.run_frame()
        elif self.game_wrapper is not None:
            self.game_wrapper.reset(self.gba)

        if self.save_episode_trajectory:
            num_episodes = len(list(Path(self.episode_trajectory_path).glob("episode_*")))
            self._curr_trajectory_path = Path(self.episode_trajectory_path) / f"episode_{num_episodes:04d}"

        observation = self._get_observation()
        
        info = {}
        if self.game_wrapper is not None:
            # self.game_wrapper.reset(self.gba)
            info.update(self.game_wrapper.info(self.gba, observation))
        return observation, info
