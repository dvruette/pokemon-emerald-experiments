import os
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
from pygba import PyGBAEnv, PyGBA
from pygba.utils import KEY_MAP

from emerald_wrapper import CustomEmeraldWrapper


class EmeraldEnv(PyGBAEnv):
    def __init__(
        self,
        gba: PyGBA,
        rank: int = 0,
        frames_path: str | Path | None = None,
        save_episode_frames: bool = False,
        frame_save_freq: int = 1,
        early_stopping: bool = False,
        patience: int = 1024,
        early_stopping_penalty: float = 0.0,
        action_noise: float = 0.0,
        reset_to_new_game_prob: float = 1.0,
        save_intermediate_state_prob: float = 0.001,
        **kwargs,
    ):
        game_wrapper = CustomEmeraldWrapper()
        self._intermediate_state = None
        super().__init__(gba, game_wrapper, **kwargs)
        self.rank = rank
        self.save_episode_frames = save_episode_frames
        self.frames_path = frames_path
        self.frame_save_freq = frame_save_freq
        self.early_stopping = early_stopping
        self.patience = patience
        self.early_stopping_penalty = early_stopping_penalty
        self.action_noise = action_noise
        self.reset_to_new_game_prob = reset_to_new_game_prob
        self.save_intermediate_state_prob = save_intermediate_state_prob

        self.arrow_keys = [None, "up", "down", "right", "left"]
        # self.buttons = [None, "A", "B", "select", "start", "L", "R"]
        self.buttons = [None, "A", "B"]

        # cartesian product of arrows and buttons, i.e. can press 1 arrow and 1 button at the same time
        self.actions = [(a, b) for a in self.arrow_keys for b in self.buttons]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        
        self._total_reward = 0
        self._max_reward = 0
        self._max_reward_step = 0
        self.agent_stats = []

    def step(self, action_id):
        info = {}

        if np.random.random() < self.action_noise:
            action_id = np.random.randint(len(self.actions))
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

        if np.random.random() > self.repeat_action_probability:
            self.gba.core.set_keys(*actions)

        if isinstance(self.frameskip, tuple):
            frameskip = np.random.randint(*self.frameskip)
        else:
            frameskip = self.frameskip

        for _ in range(frameskip + 1):
            self.gba.core.run_frame()
            pass
        observation = self._get_observation()

        # save intermediate state
        if np.random.random() < self.save_intermediate_state_prob:
            self._intermediate_state = self.gba.core.save_raw_state()

        # reward calculation
        reward = 0
        if self.game_wrapper is not None:
            reward = self.game_wrapper.reward(self.gba, observation)
            info.update(self.game_wrapper.info(self.gba, observation))
        if self.early_stopping and self._step - self._max_reward_step > self.patience:
            reward -= self.early_stopping_penalty

        self._total_reward += reward
        if self._total_reward > self._max_reward:
            self._max_reward = self._total_reward
            self._max_reward_step = self._step
        
        info["rewards"]["total_reward"] = self._total_reward

        # the tensorboard will read out the agent_stats list and plot it
        self.agent_stats.append(info["rewards"])

        done = self.check_if_done()
        truncated = self.check_if_truncated()

        self._step += 1
        reward_display = " | ".join(f"{k}={v:.3g}" for k, v in info["rewards"].items())
        print(f"\r step={self._step:5d} | {reward_display}", end="", flush=True)
        return observation, reward, done, truncated, info
    
    def check_if_truncated(self):
        if self.max_episode_steps is not None and self._step >= self.max_episode_steps:
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._total_reward = 0
        self._max_reward = 0
        self._max_reward_step = 0
        self.agent_stats = []

        if self._intermediate_state is not None and np.random.random() >= self.reset_to_new_game_prob:
            self.gba.core.load_raw_state(self._intermediate_state)
            self.gba.core.run_frame()

        observation = self._get_observation()
        
        info = {}
        if self.game_wrapper is not None:
            self.game_wrapper.reset(self.gba)
            info.update(self.game_wrapper.info(self.gba, observation))
        return observation, info
