import gym
import numpy as np


class DummyOffsetWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DummyOffsetWrapper, self).__init__(env)
        self.observation_space = self.get_new_observation_space(self.observation_space)

    def get_stacked_observation_space(self, observation_space):
        assert isinstance(observation_space, gym.spaces.Dict)
        new_observation_space = {}
        for k, v in self.observation_space.spaces.items():
            new_observation_space[k] = v
        new_observation_space['u_shift'] = gym.spaces.Box(0.0, 0.16, shape=(2,))
        new_observation_space['v_shift'] = gym.spaces.Box(0.0, 0.16, shape=(2,))
        new_observation_space = gym.spaces.Dict(new_observation_space)

        return new_observation_space

    def observation(self, observation):
        observation['u_shift'] = np.zeros((2,))
        observation['v_shift'] = np.zeros((2,))

        return observation
