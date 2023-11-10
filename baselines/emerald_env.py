import os

import gymnasium as gym
import numpy as np
from pygba import PyGBAEnv, PyGBA
from pygba.utils import KEY_MAP

from emerald_wrapper import CustomEmeraldWrapper


class EmeraldEnv(PyGBAEnv):
    def __init__(
        self,
        gba: PyGBA,
        **kwargs,
    ):
        game_wrapper = CustomEmeraldWrapper()
        super().__init__(gba, game_wrapper, **kwargs)

        self.arrow_keys = [None, "up", "down", "right", "left"]
        # self.buttons = [None, "A", "B", "select", "start", "L", "R"]
        self.buttons = [None, "A", "B"]

        # cartesian product of arrows and buttons, i.e. can press 1 arrow and 1 button at the same time
        self.actions = [(a, b) for a in self.arrow_keys for b in self.buttons]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        
        self._total_reward = 0

    def step(self, action_id):
        info = {}

        actions = self.get_action_by_id(action_id)
        actions = [KEY_MAP[a] for a in actions if a is not None]

        os.makedirs("frames", exist_ok=True)
        img = self._framebuffer.to_pil().convert("RGB")
        img.save(f"frames/{self.rank:02d}.png")

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

        reward = 0
        done = False
        truncated = False
        if self.max_episode_steps is not None:
            truncated = self._step >= self.max_episode_steps
        if self.game_wrapper is not None:
            reward = self.game_wrapper.reward(self.gba, observation)
            done = done or self.game_wrapper.game_over(self.gba, observation)
            info.update(self.game_wrapper.info(self.gba, observation))

        self._total_reward += reward
        self._step += 1
        print(f"\r step={self._step} | {reward=} | total_reward={self._total_reward} | {done=} | {truncated=}", end="", flush=True)

        return observation, reward, done, truncated, info
    
    def check_if_done(self):
        done = False
        if self.game_wrapper is not None:
            observation = self._get_observation()
            done = self.game_wrapper.game_over(self.gba, observation)
        return done

    def reset(self, seed=None):
        self._total_reward = 0
        return super().reset(seed=seed)
