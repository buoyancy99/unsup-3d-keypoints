from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from stable_baselines3.common.preprocessing import preprocess_obs

import gym
import torch as th
import torch.nn as nn
from .augmentation import center_crop_image
from .extractors import AugmentHybridCnnExtractor
from .utils import preprocess_obs


class AugmentPolicyMixin(object):
    def __init__(self, *args, augment=False, offset_crop=False, first_frame=None, **kwargs):
        self.augment = augment
        self.crop_size = None
        self.offset_crop = offset_crop
        self.first_frame = first_frame
        if self.augment:
            observation_space = kwargs['observation_space'] if 'observation_space' in kwargs else args[0]
            self.calculate_crop_size(observation_space)
        if 'features_extractor_kwargs' in kwargs and \
                issubclass(kwargs['features_extractor_class'], AugmentHybridCnnExtractor):
            # sac's actor and critic policy alone don't have features_extractor_kwargs
            kwargs['features_extractor_kwargs'].update(crop_size=self.crop_size)
        super(AugmentPolicyMixin, self).__init__(*args, **kwargs)

    def calculate_crop_size(self, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            self.crop_size = round(observation_space['images'].shape[-2] * 0.84)
        elif isinstance(observation_space, gym.spaces.Tuple):
            raise NotImplementedError
        else:
            self.crop_size = round(observation_space.shape[-2] * 0.84)

        if self.crop_size % 2:
            self.crop_size += 1

    def center_crop(self, images):
        """
        :param images: [..., h, w, c]
        :return: cropped_images: [..., crop_size, crop_size, c]
        """
        assert self.augment, 'Only do center crop when policy has augmentation'
        return center_crop_image(images, self.crop_size)

    def process_uncropped_obs(self, obs):
        if self.augment:
            if isinstance(self.observation_space, gym.spaces.Dict):
                obs = obs.copy()
                obs['images'] = self.center_crop(obs['images'])
                if self.offset_crop:
                    frame_stack, num_cameras, _, original_size, _ = self.observation_space['images'].shape
                    batch_size = len(obs['images'])
                    shift = (original_size - self.crop_size) / original_size / 2.0
                    obs['u_shift'] = th.ones((batch_size, frame_stack, num_cameras)).to(self.device) * shift
                    obs['v_shift'] = th.ones((batch_size, frame_stack, num_cameras)).to(self.device) * shift
                if self.first_frame is not None:
                    first_frame = th.from_numpy(self.first_frame).to(self.device)
                    cropped_first_frame = self.center_crop(first_frame)
                    cropped_first_frame = cropped_first_frame[None, None].expand_as(obs['images'])
                    obs['first_frame'] = cropped_first_frame
            elif isinstance(self.observation_space, gym.spaces.Tuple):
                raise NotImplementedError
            else:
                obs = self.center_crop(obs)
        elif self.first_frame is not None:
            first_frame = th.from_numpy(self.first_frame).to(self.device)
            first_frame = first_frame[None, None].expand_as(obs['images'])
            obs['first_frame'] = first_frame

        return obs

    def _get_data(self) -> Dict[str, Any]:
        data = super(AugmentPolicyMixin, self)._get_data()
        data.update(
            augment=self.augment
        )
        return data


class UnsupPolicyMixin(object):
    def _encode_obs(self, obs: Union[th.Tensor, Dict, Tuple]) -> th.Tensor:
        raise NotImplementedError

    def extract_features(self, obs: Union[th.Tensor, Dict, Tuple], detach_encoder) -> th.Tensor:
        assert self.features_extractor is not None, 'No feature extractor was set'
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)

        if detach_encoder:
            with th.no_grad():
                unsup_net_out = self._encode_obs(obs)
        else:
            unsup_net_out = self._encode_obs(obs)

        image_feature, unsup_loss_dict = unsup_net_out
        obs['image_feature'] = image_feature
        features = self.features_extractor(obs)

        return features, unsup_net_out

    def _get_data(self) -> Dict[str, Any]:
        data = super(UnsupPolicyMixin, self)._get_data()
        data.update(
            unsup_net_class=self.unsup_net_class,
            unsup_net_kwargs=self.unsup_net_kwargs,
            unsup_coef_dict=self.unsup_coef_dict
        )
        return data

    def load_unsup_net(self, path):
        print('Load unsup net from {}'.format(path))
        checkpoint = th.load(path)
        self.unsup_net.load_state_dict(checkpoint['unsup_net'])
        self.unsup_optimizer.load_state_dict(checkpoint['unsup_optimizer'])

    def save_unsup_net(self, path):
        print('Save unsup net to {}'.format(path))
        th.save(dict(
            unsup_net=self.unsup_net.state_dict(),
            unsup_optimizer=self.unsup_optimizer.state_dict()
        ))


class MultiviewPolicyMixin(object):
    def __init__(self, *args, latent_stack=False, **kwargs):
        self.latent_stack = latent_stack
        super(MultiviewPolicyMixin, self).__init__(*args, **kwargs)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.frame_stack = self.observation_space['images'].shape[0]
        else:
            self.frame_stack = self.observation_space.shape[0]

    def flatten_multiview_images(self, images):
        """
        :param images: [batch, frame_stack, num_cameras, h, w ,c]
        :return: processed_images: []
        """
        batch, frame_stack, num_cameras, h, w, c = images.shape
        if self.latent_stack:
            processed_images = images.permute(0, 1, 2, 5, 3, 4)
            processed_images = processed_images.reshape(batch * frame_stack, num_cameras * c, h, w)
        else:
            processed_images = images.permute(0, 2, 1, 5, 3, 4)
            processed_images = processed_images.reshape(batch, num_cameras * frame_stack * c, h, w)

        return processed_images

    def unflatten_multiview_feature(self, feature):
        if self.latent_stack:
            batch, feature_shape = feature.shape[0], feature.shape[1:]
            feature = feature.view(batch // self.frame_stack, self.frame_stack, *feature_shape)
            feature = th.cat([feature[:, -1:], feature[:, 1:] - feature[:, :-1]], 1)
            return feature
        else:
            return feature[:, None]

    def flatten_multiview_offset(self, offset):
        batch, frame_stack, num_cameras = offset.shape
        if self.latent_stack:
            offset = offset.view(batch * frame_stack, num_cameras)
        else:
            offset = offset[:, 0]
        return offset

    def process_multiview_obs(self, obs):
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs = obs.copy()
            obs['images'] = self.flatten_multiview_images(obs['images'])
            if self.offset_crop:
                obs['u_shift'] = self.flatten_multiview_offset(obs['u_shift'])
                obs['v_shift'] = self.flatten_multiview_offset(obs['v_shift'])
            if self.first_frame is not None:
                obs['first_frame'] = self.flatten_multiview_images(obs['first_frame'])
        elif isinstance(self.observation_space, gym.spaces.Tuple):
            raise NotImplementedError
        else:
            obs = self.flatten_multiview_images(obs)

        return obs

    def process_multiview_feature(self, obs, feature):
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs = obs.copy()
            obs['image_feature'] = self.unflatten_multiview_feature(feature)
        elif isinstance(self.observation_space, gym.spaces.Tuple):
            raise NotImplementedError
        else:
            obs = self.unflatten_multiview_feature(obs)

        return obs


class KeypointPolicyMixin(MultiviewPolicyMixin, UnsupPolicyMixin):
    def ae_criteria(self, images_hat, images):
        images = self.unsup_net.upsampler(images)
        return nn.functional.mse_loss(images_hat, images)

    def _encode_obs(self, obs):
        u_shift = obs['u_shift'] if self.offset_crop else None
        v_shift = obs['v_shift'] if self.offset_crop else None
        return self.unsup_net.encode(obs['images'], rsample=False, u_shift=u_shift, v_shift=v_shift)

    def extract_features(self, obs: Union[th.Tensor, Dict, Tuple], detach_encoder) -> th.Tensor:
        assert self.features_extractor is not None, 'No feature extractor was set'
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)

        if detach_encoder:
            with th.no_grad():
                unsup_net_out = self._encode_obs(obs)
        else:
            unsup_net_out = self._encode_obs(obs)

        image_feature, unsup_loss_dict, _ = unsup_net_out
        # image_feature = image_feature[:, :, :-1]
        obs = self.process_multiview_feature(obs, image_feature)
        features = self.features_extractor(obs)

        return features, unsup_net_out
