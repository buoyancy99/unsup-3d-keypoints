import pickle
import gym.spaces as spaces
import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from typing import Tuple

import numpy as np


class MultiTaskRunningMeanStd(RunningMeanStd):
    def update(self, arr: np.ndarray) -> None:
        assert arr.shape == self.mean.shape
        delta = arr - self.mean
        tot_count = self.count + 1

        new_mean = self.mean + delta / tot_count
        m_a = self.var * self.count
        m_b = 1.0 * 1
        m_2 = m_a + m_b + np.square(delta) * self.count / (self.count + 1)
        new_var = m_2 / (self.count + 1)

        new_count = 1 + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class MultiTaskNormalize(VecNormalize):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(
            self, venv, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            epsilon=1e-8
    ):
        VecEnvWrapper.__init__(self, venv)
        if isinstance(self.observation_space, spaces.Dict):
            self.obs_rms = {RunningMeanStd(shape=space.shape) for k, space in self.observation_space.spaces.items()}
        elif isinstance(self.observation_space, spaces.Tuple):
            self.obs_rms = tuple(RunningMeanStd(shape=space.shape) for space in self.observation_space.spaces)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = MultiTaskRunningMeanStd(shape=(self.num_envs, ))
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])
        self.old_reward = np.array([])

    def normalize_reward(self, reward, task_ids):
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var[task_ids] + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward

    def unnormalize_reward(self, reward, task_ids):
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var[task_ids] + self.epsilon)
        return reward
