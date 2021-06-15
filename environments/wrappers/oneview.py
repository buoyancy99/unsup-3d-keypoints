import gym
import numpy as np


class OneviewWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneviewWrapper, self).__init__(env)

    def observation(self, observation):
        observation['images'][0] = 0
        observation['images'][2] = 0

        return observation
