from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import torch as th
from torch import nn as nn

from ..common.policy_mixins import MultiviewPolicyMixin
from ..common.ppo.ppo_base_policy import AugmentPpoPolicy
from ..common.extractors import AugmentHybridCnnExtractor


class MultiviewAugmentPpoPolicy(MultiviewPolicyMixin, AugmentPpoPolicy):
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
        features_extractor_class: Type[BaseFeaturesExtractor] = AugmentHybridCnnExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        augment: bool = False,
    ):

        super(MultiviewAugmentPpoPolicy, self).__init__(
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
            augment=augment
        )

    def _get_latent(self, obs: Union[th.Tensor, Dict, Tuple]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = self.process_multiview_obs(obs)

        return super(MultiviewAugmentPpoPolicy, self)._get_latent(obs)

