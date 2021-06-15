from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

import gym
import torch as th
import torch.nn as nn
import numpy as np

from ..policy_mixins import AugmentPolicyMixin, UnsupPolicyMixin


class AugmentPpoPolicy(AugmentPolicyMixin, ActorCriticPolicy):
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.augment:
            obs = self.process_uncropped_obs(obs)
        return super(AugmentPpoPolicy, self).forward(obs)

    def _predict(self, observation: Union[th.Tensor, Dict, Tuple], deterministic: bool = False) -> th.Tensor:
        if self.augment:
            observation = self.process_uncropped_obs(observation)
        return super(AugmentPpoPolicy, self)._predict(observation)


class UnsupPpoPolicy(UnsupPolicyMixin, AugmentPpoPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        unsup_net_class=None,
        unsup_net_kwargs: dict = {},
        train_jointly: bool = False,
        offset_crop: bool = False,
        first_frame: np.ndarray = None,
    ):
        self.unsup_net_class = unsup_net_class
        self.unsup_net_kwargs = unsup_net_kwargs
        self.train_jointly = train_jointly

        super(UnsupPpoPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            augment=augment,
            offset_crop=offset_crop,
            first_frame=first_frame
        )

    def _get_data(self) -> Dict[str, Any]:
        data = super(UnsupPpoPolicy, self)._get_data()
        data.update(
            train_jointly=self.train_jointly,
            offset_crop=self.offset_crop,
            first_frame=self.first_frame
        )
        return data

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super(UnsupPpoPolicy, self)._build(lr_schedule)
        self.params_exclude_unsup = list(self.parameters())
        self.unsup_net = self.unsup_net_class(self.observation_space, self.crop_size, **self.unsup_net_kwargs)
        self.unsup_optimizer = self.optimizer_class(
            self.unsup_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

        self.optimizer = self.optimizer_class(
            self.params_exclude_unsup + list(self.unsup_net.parameters()) * int(self.train_jointly),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        self.features_extractor.eval()
        obs = self.process_uncropped_obs(obs)
        latent_pi, latent_vf, latent_sde, _ = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: Union[th.Tensor, Dict, Tuple]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: (th.Tensor) Observation
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features, unsup_net_out = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde, unsup_net_out

    def _predict(self, observation: Union[th.Tensor, Dict, Tuple], deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        self.features_extractor.eval()
        if self.augment:
            observation = self.process_uncropped_obs(observation)
        latent_pi, _, latent_sde, _ = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: Union[th.Tensor, Dict, Tuple],
                         actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde, unsup_net_out = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        _, unsup_loss_dict = unsup_net_out

        return values, log_prob, distribution.entropy(), unsup_loss_dict
