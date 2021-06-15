from typing import Tuple, Union, Dict

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


def preprocess_obs(obs: Union[th.Tensor, Dict, Tuple], observation_space: spaces.Space,
                   normalize_images: bool = True, allow_unexpected: bool = True) -> th.Tensor:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.
    :param obs: (th.Tensor) Observation
    :param observation_space: (spaces.Space)
    :param normalize_images: (bool) Whether to normalize images or not
        (True by default)
    :param allow_unexpected: allow keys that's not present in observation space, for dict obs only
    :return: (th.Tensor)
    """
    if isinstance(observation_space, spaces.Box):
        if observation_space.dtype == np.uint8 and normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        processed_obs = {}
        for k, o in obs.items():
            if k in observation_space.spaces:
                processed_obs[k] = preprocess_obs(o, observation_space.spaces[k], normalize_images)
            elif allow_unexpected:
                if o.dtype == th.uint8:
                    o = o / 255.0
                processed_obs[k] = o.float()
            else:
                raise AttributeError('key {} not in observation space, set allow_unexpected=True to override'.format(k))

        return processed_obs

    elif isinstance(observation_space, spaces.Tuple):
        return tuple(preprocess_obs(o, os, normalize_images) for o, os in zip(obs, observation_space.spaces))

    else:
        raise NotImplementedError()