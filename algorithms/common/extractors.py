import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple
import numpy as np
from .models.cnn import NatureEncoder


class HybridFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 512,
        state_feature_keys: list = [],
    ):
        self.state_feature_keys = state_feature_keys
        super(HybridFeatureExtractor, self).__init__(observation_space, features_dim)

        state_featured_dim = 0
        for k in state_feature_keys:
            state_featured_dim += int(np.prod(self._observation_space[k].shape))
        self.state_featured_dim = state_featured_dim


class AugmentHybridCnnExtractor(HybridFeatureExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 512,
        crop_size: int = None,
        encoder_cls: BaseFeaturesExtractor = NatureEncoder,
        encoder_kwargs: dict = {},
        state_feature_keys: list = [],
    ):
        super(AugmentHybridCnnExtractor, self).__init__(
            observation_space,
            features_dim=features_dim,
            state_feature_keys=state_feature_keys
        )
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        encoder_default_kwargs = dict(out_channels=32, n_filters=32, groups=1)
        encoder_default_kwargs.update(encoder_kwargs)
        encoder_kwargs = encoder_default_kwargs
        image_space = observation_space['images']
        in_channels = int(np.prod(image_space.shape) / np.prod(image_space.shape[-3:-1]))

        self.cnn = encoder_cls(in_channels, **encoder_kwargs)
        self.flatten = nn.Flatten()

        input_size = image_space.shape[-2] if crop_size is None else crop_size
        output_size = self.cnn.infer_output_size(input_size)
        image_feature_dim = encoder_kwargs['out_channels'] * output_size ** 2

        self.linear = nn.Sequential(nn.Linear(image_feature_dim + self.state_featured_dim, features_dim), nn.ReLU())

    def forward(self, observations: [th.Tensor, Dict, Tuple]) -> th.Tensor:
        images = observations['images']
        image_feature = self.flatten(self.cnn(images))
        if self.state_feature_keys:
            state_feature = [v.flatten(1) for k, v in observations.items() if k in self.state_feature_keys]
            latent = th.cat([image_feature] + state_feature, 1)
        else:
            latent = image_feature

        return self.linear(latent)


class UnsupHybridExtractor(HybridFeatureExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 512,
        image_feature_dim: int = None,
        renormalize: bool = False,
        state_feature_keys: list = []
    ):
        super(UnsupHybridExtractor, self).__init__(
            observation_space,
            features_dim=features_dim,
            state_feature_keys=state_feature_keys)

        assert image_feature_dim is not None, 'Must pass in image_feature_dim'
        self.image_feature_dim = image_feature_dim
        self.renormalize = renormalize

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            # nn.BatchNorm1d(image_feature_dim + self.state_featured_dim, eps=1e-2, momentum=.05) if renormalize else nn.Identity(),
            nn.Linear(image_feature_dim + self.state_featured_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: [th.Tensor, Dict, Tuple]) -> th.Tensor:
        image_feature = observations['image_feature']
        image_feature = self.flatten(image_feature)
        if self.state_feature_keys:
            state_feature = [v.flatten(1) for k, v in observations.items() if k in self.state_feature_keys]
            latent = th.cat([image_feature] + state_feature, 1)
        else:
            latent = image_feature

        return self.linear(latent)


class KeypointHybridExtractor(UnsupHybridExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 512,
        num_keypoints: int = None,
        keypoint_dim: int = 4,
        renormalize: bool = True,
        state_feature_keys: list = []
    ):
        assert num_keypoints is not None, 'Must pass in num of keypoints'

        self.num_keypoints = num_keypoints
        self.temporal_dim = observation_space['images'].shape[0]
        self.keypoint_dim = keypoint_dim
        image_feature_dim = num_keypoints * (keypoint_dim + 1) * self.temporal_dim

        super(KeypointHybridExtractor, self).__init__(
            observation_space,
            features_dim=features_dim,
            image_feature_dim=image_feature_dim,
            renormalize=renormalize,
            state_feature_keys=state_feature_keys,
        )

    def forward(self, observations: [th.Tensor, Dict, Tuple]) -> th.Tensor:
        image_feature = observations['image_feature']

        if self.temporal_dim > 1:
            latest, diff = th.split(image_feature, [1, self.temporal_dim - 1], dim=1)

            keypoint, confidence = th.split(latest, [self.keypoint_dim, 1], dim=-1)
            confidence = (confidence * self.num_keypoints - 1.0) * 2.0
            latest = th.cat([keypoint, confidence], -1)

            image_feature = th.cat([latest, diff * 10.0], 1)

        else:
            keypoint, confidence = th.split(image_feature, [self.keypoint_dim, 1], dim=-1)
            confidence = (confidence * self.num_keypoints - 1.0) * 2.0
            image_feature = th.cat([keypoint, confidence], -1)

        image_feature = self.flatten(image_feature)

        if self.state_feature_keys:
            state_feature = [v.flatten(1) for k, v in observations.items() if k in self.state_feature_keys]
            latent = th.cat([image_feature] + state_feature, 1)
        else:
            latent = image_feature

        return self.linear(latent)
