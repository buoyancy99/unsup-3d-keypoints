import gym
import numpy as np


class FrameStackWrapper(gym.ObservationWrapper):
    r"""Upsample the image observation to a square image. """

    def __init__(self, env, frame_stack=1):
        super(FrameStackWrapper, self).__init__(env)
        self.frame_stack = frame_stack
        self.obs_buffer = None
        self.observation_space = self.get_stacked_observation_space(self.observation_space)

    def get_stacked_observation_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            new_observation_space = {}
            for k, v in self.observation_space.spaces.items():
                new_observation_space[k] = self.get_stacked_observation_space(v)
            new_observation_space = gym.spaces.Dict(new_observation_space)
        elif isinstance(observation_space, gym.spaces.Tuple):
            new_observation_space = []
            for v in self.observation_space.spaces:
                new_observation_space.append(self.get_stacked_observation_space(v))
            new_observation_space = gym.spaces.Tuple(new_observation_space)
        else:
            low, high, dtype = observation_space.low, observation_space.high, observation_space.dtype
            low = np.repeat(low[None], self.frame_stack, axis=0)
            high = np.repeat(high[None], self.frame_stack, axis=0)
            new_observation_space = observation_space.__class__(low=low, high=high, dtype=dtype)

        return new_observation_space

    def reset(self, **kwargs):
        self.reset_obs_buffer()
        observation = super(FrameStackWrapper, self).reset(**kwargs)
        return observation

    def reset_obs_buffer(self):
        self.obs_buffer = self.observation_space.sample()
        self.obs_buffer = self._update_obs_buffer(self.obs_buffer)

    def _update_obs_buffer(self, obs_buffer, new_obs=None):
        if isinstance(obs_buffer, np.ndarray):
            if new_obs is None:
                obs_buffer[:] = 0
            else:
                obs_buffer[:-1] = obs_buffer[1:]
                obs_buffer[-1] = new_obs
        elif isinstance(obs_buffer, dict):
            obs_buffer = {k: self._update_obs_buffer(v, None if new_obs is None else new_obs[k])
                          for k, v in obs_buffer.items()}
        elif isinstance(obs_buffer, list) or isinstance(obs_buffer, tuple):
            obs_buffer = [self._update_obs_buffer(obs_buffer[i], None if new_obs is None else new_obs[i])
                          for i in range(len(obs_buffer))]
        else:
            raise NotImplementedError

        return obs_buffer

    def observation(self, observation):
        if self.obs_buffer is None:
            raise RuntimeError("Must reset obs buffer first")
        self.obs_buffer = self._update_obs_buffer(self.obs_buffer, observation)

        return self.obs_buffer
