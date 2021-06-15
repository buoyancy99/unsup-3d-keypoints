from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import torch as th
from torch import nn as nn
import numpy as np

from ..common.policy_mixins import KeypointPolicyMixin
from ..common.ppo.ppo_base_policy import UnsupPpoPolicy
from ..common.extractors import KeypointHybridExtractor
from ..common.models.keypoint_net import KeypointNetBase
from ..common.utils import preprocess_obs


class MultiviewKeypointPpoPolicy(KeypointPolicyMixin, UnsupPpoPolicy):
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
        features_extractor_class: Type[BaseFeaturesExtractor] = KeypointHybridExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        unsup_net_class: KeypointNetBase = None,
        unsup_net_kwargs: dict = {},
        train_jointly: bool = True,
        latent_stack: bool = False,
        offset_crop: bool = False,
        first_frame: np.ndarray = None,
    ):
        unsup_net_kwargs.update(latent_stack=latent_stack)

        super(MultiviewKeypointPpoPolicy, self).__init__(
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
            latent_stack=latent_stack,
            augment=augment,
            unsup_net_class=unsup_net_class,
            unsup_net_kwargs=unsup_net_kwargs,
            train_jointly=train_jointly,
            offset_crop=offset_crop,
            first_frame=first_frame
        )

    def _get_data(self) -> Dict[str, Any]:
        data = super(MultiviewKeypointPpoPolicy, self)._get_data()
        data.update(
            latent_stack=self.latent_stack
        )
        return data

    def _get_latent(self, obs: Union[th.Tensor, Dict, Tuple]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = self.process_multiview_obs(obs)

        return super(MultiviewKeypointPpoPolicy, self)._get_latent(obs)

    def extract_features(self, obs):
        return super(MultiviewKeypointPpoPolicy, self).extract_features(obs, not self.train_jointly)

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
        keypoints, unsup_loss_dict, _ = unsup_net_out

        if self.train_jointly:
            obs = preprocess_obs(self.process_multiview_obs(obs), self.observation_space)
            u_shift = obs['u_shift'] if self.offset_crop else None
            v_shift = obs['v_shift'] if self.offset_crop else None
            first_frame = obs['first_frame'] if self.first_frame is not None else None
            images_hat = self.unsup_net.decode(keypoints, u_shift=u_shift, v_shift=v_shift, first_frame=first_frame)
            unsup_loss_dict['ae_loss'] = self.ae_criteria(images_hat, obs['images'])

        return values, log_prob, distribution.entropy(), unsup_loss_dict

    def visualize(self, obs):
        batch, frame_stack, num_cameras, _, _, c = obs['images'].shape
        obs = self.process_multiview_obs(obs)
        obs = preprocess_obs(obs, self.observation_space)
        images = obs['images']
        u_shift = obs['u_shift'] if self.offset_crop else None
        v_shift = obs['v_shift'] if self.offset_crop else None
        first_frame = obs['first_frame'] if self.first_frame is not None else None

        keypoints, _, heatmap = self.unsup_net.encode(images, rsample=False, u_shift=u_shift, v_shift=v_shift)

        images_hat = self.unsup_net.decode(keypoints, u_shift=u_shift, v_shift=v_shift, first_frame=first_frame)
        images = self.unsup_net.upsampler(images)
        h, w = images_hat.shape[-2:]

        if self.latent_stack:
            keypoints = self.unflatten_multiview_feature(keypoints)[:, 0]
            heatmap = self.unflatten_multiview_feature(heatmap)[:, 0]
            if u_shift is not None:
                u_shift = u_shift.view(batch, frame_stack, num_cameras)[:, -1]
                v_shift = v_shift.view(batch, frame_stack, num_cameras)[:, -1]
            images = images.view(batch, frame_stack, num_cameras, c, h, w)[:, -1]
            images_hat = images_hat.view(batch, frame_stack, num_cameras, c, h, w)[:, -1]
        else:
            images = images.view(batch, num_cameras, frame_stack, c, h, w)[:, :, -1]
            images_hat = images_hat.view(batch, num_cameras, frame_stack, c, h, w)[:, :, -1]

        vis_tensor = self.unsup_net.visualize(images, keypoints, images_hat, heatmap, u_shift, v_shift)

        return vis_tensor

