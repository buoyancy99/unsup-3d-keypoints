from typing import Union, Optional
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from .augmentation import random_crop
from gym import spaces
import numpy as np
import torch as th
from threading import Thread
from queue import Queue
import time


class BufferAugmentMixin(object):
    def __init__(self, crop_size, *args, pass_offset=False, first_frame=None, **kwargs):
        super(BufferAugmentMixin, self).__init__(*args, **kwargs)
        self.crop_size = crop_size
        self.pass_offset = pass_offset
        self.first_frame = first_frame
        self.cache = Queue(maxsize=1)
        self.cache_batch_size = None
        self.worker = None
        assert not isinstance(self.observation_space, spaces.Tuple), \
            'Tuple obs not supported for data augmentation buffer, use Dict obs '

    def augment_obs(self, observation_samples):
        images = observation_samples['images']
        image_size = images.shape[-2]
        batch_size = images.shape[0]

        if self.crop_size is not None:
            crop_max = image_size - self.crop_size
            if len(images.shape) > 4:
                # need same crop across frame stack
                batch_size, frame_stack, extra_dims = images.shape[0], images.shape[1], images.shape[2:-3]
                r_vec = np.random.randint(0, crop_max, (batch_size, 1, *extra_dims)).repeat(frame_stack, 1)
                c_vec = np.random.randint(0, crop_max, (batch_size, 1, *extra_dims)).repeat(frame_stack, 1)
            else:
                r_vec = np.random.randint(0, crop_max, batch_size)
                c_vec = np.random.randint(0, crop_max, batch_size)

            v_shift = r_vec / image_size
            u_shift = c_vec / image_size

            batch_shape = images.shape[:-3]
            images = images.reshape(-1, *images.shape[-3:])
            cropped_images = random_crop(images, self.crop_size, r_vec=r_vec.flatten(), c_vec=c_vec.flatten())
            cropped_images = cropped_images.reshape(*batch_shape, *cropped_images.shape[-3:])

            observation_samples['images'] = cropped_images

            if self.pass_offset:
                observation_samples['u_shift'] = u_shift
                observation_samples['v_shift'] = v_shift

            if self.first_frame is not None:
                first_frame = self.first_frame[None].repeat(batch_size, 0).reshape(-1, *images.shape[-3:])
                cropped_first_frame = random_crop(first_frame, self.crop_size,
                                                  r_vec=r_vec[:, 0].flatten(), c_vec=c_vec[:, 0].flatten())
                cropped_first_frame = cropped_first_frame.reshape(batch_size, 1, *extra_dims, *cropped_images.shape[-3:])
                cropped_first_frame = cropped_first_frame.repeat(frame_stack, 1)
                observation_samples['first_frame'] = cropped_first_frame

        elif self.first_frame is not None:
            if len(images.shape) > 4:
                # need same crop across frame stack
                batch_size, frame_stack, extra_dims = images.shape[0], images.shape[1], images.shape[2:-3]
            first_frame = self.first_frame[None].repeat(batch_size, 0)[:, None].repeat(frame_stack, 1)
            observation_samples['first_frame'] = first_frame

        return observation_samples



    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        if self.worker is None:
            self.cache_batch_size = batch_size
            self.worker = Thread(target=self.sample_inf, kwargs=dict(env=env), daemon=True)
            self.worker.start()

        if self.cache_batch_size != batch_size:
            self.cache_batch_size = batch_size
            _ = self.cache.get()

        samples = self.cache.get()

        self.cache_batch_size = batch_size
        return samples

    def sample_inf(self, env: Optional[VecNormalize] = None):
        while True:
            if self.cache.empty():
                sample = super(BufferAugmentMixin, self).sample(batch_size=self.cache_batch_size, env=env)
                self.cache.put(sample)
            else:
                time.sleep(0.005)


class AugmentRolloutBuffer(BufferAugmentMixin, RolloutBuffer):
    def __init__(
        self,
        crop_size: int,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        pass_offset: bool = False,
        first_frame: np.ndarray = None,
        device: Union[th.device, str] = 'cpu',
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1
    ):
        super(AugmentRolloutBuffer, self).__init__(
            crop_size,
            buffer_size,
            observation_space,
            action_space,
            pass_offset=pass_offset,
            first_frame=first_frame,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs
        )

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {}
            for k, obs in self.observations.items():
                observation_samples[k] = obs[batch_inds]
        elif isinstance(self.observation_space, spaces.Tuple):
            for obs in self.observations:
                observation_samples = obs[batch_inds]
        else:
            observation_samples = self.observations[batch_inds]

        observation_samples = self.augment_obs(observation_samples)

        data = (observation_samples,
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class AugmentObsBuffer(BufferAugmentMixin, BaseBuffer):
    def __init__(
        self,
        crop_size: int,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        pass_offset: bool = False,
        first_frame: np.ndarray = None,
        device: Union[th.device, str] = "cpu"
    ):
        super(AugmentObsBuffer, self).__init__(
            crop_size,
            buffer_size,
            observation_space,
            action_space,
            pass_offset=pass_offset,
            first_frame=first_frame,
            device=device,
            n_envs=1,
        )

        if isinstance(self.observation_space, spaces.Dict):
            self.observations = {k: np.zeros((self.buffer_size,) + obs_shape, dtype=self.obs_dtype[k])
                                 for k, obs_shape in self.obs_shape.items()}
        elif isinstance(self.observation_space, spaces.Tuple):
            self.observations = tuple(np.zeros((self.buffer_size,) + obs_shape, dtype=obs_dtype)
                                      for obs_shape, obs_dtype in zip(self.obs_shape, self.obs_dtype))
        else:
            self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=self.obs_dtype)

    def add(self, obs: Union[np.ndarray, dict, tuple]) -> None:
        if isinstance(self.observation_space, spaces.Dict):
            for k, obs_array in obs.items():
                cap = min(self.buffer_size - self.pos, len(obs_array))
                self.observations[k][self.pos:self.pos + cap] = np.array(obs_array)[:cap]
            self.pos += len(obs_array)
        elif isinstance(self.observation_space, spaces.Tuple):
            for i, obs_array in enumerate(obs):
                cap = min(self.buffer_size - self.pos, len(obs_array))
                self.observations[i][self.pos:self.pos + cap] = np.array(obs_array)[:cap]
            self.pos += len(obs_array)
        else:
            cap = min(self.buffer_size - self.pos, len(obs))
            self.observations[self.pos:self.pos + cap] = np.array(obs)[:cap]
            self.pos += len(obs)

        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> Union[dict, tuple, th.Tensor]:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {}
            for k, obs in self.observations.items():
                observation_samples[k] = obs[batch_inds]
        elif isinstance(self.observation_space, spaces.Tuple):
            observation_samples = []
            for obs in self.observations:
                observation_samples.append(obs[batch_inds])
            observation_samples = tuple(observation_samples)
        else:
            observation_samples = self.observations[batch_inds]

        observation_samples = self.augment_obs(observation_samples)

        return self.to_torch(observation_samples)

    def sample_cpc(self, batch_size: int, env: Optional[VecNormalize] = None):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_cpc_samples(batch_inds)

    def _get_cpc_samples(self, batch_inds: np.ndarray) -> Union[dict, tuple, th.Tensor]:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {}
            for k, obs in self.observations.items():
                observation_samples[k] = obs[batch_inds]
        elif isinstance(self.observation_space, spaces.Tuple):
            observation_samples = []
            for obs in self.observations:
                observation_samples.append(obs[batch_inds])
            observation_samples = tuple(observation_samples)
        else:
            observation_samples = self.observations[batch_inds]

        observation_samples1 = self.augment_obs(observation_samples.copy())
        observation_samples2 = self.augment_obs(observation_samples.copy())

        return self.to_torch(observation_samples1), self.to_torch(observation_samples2)


class AugmentReplayBuffer(BufferAugmentMixin, ReplayBuffer):
    def __init__(
        self,
        crop_size: int,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        pass_offset: bool = False,
        first_frame: np.ndarray = None,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False
    ):

        super(AugmentReplayBuffer, self).__init__(
            crop_size,
            buffer_size,
            observation_space,
            action_space,
            pass_offset=pass_offset,
            first_frame=first_frame,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage
        )

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {}
            next_observation_samples = {}
            for k, obs in self.observations.items():
                observation_samples[k] = self._normalize_obs(obs[batch_inds, 0], env)

            if self.optimize_memory_usage:
                for k, obs in self.observations.items():
                    next_observation_samples[k] = self._normalize_obs(
                        obs[(batch_inds + 1) % self.buffer_size, 0], env)
            else:
                for k, next_obs in self.next_observations.items():
                    next_observation_samples[k] = self._normalize_obs(next_obs[batch_inds, 0], env)

        elif isinstance(self.observation_space, spaces.Tuple):
            observation_samples = []
            next_observation_samples = []
            for obs in self.observations:
                observation_samples.append(self._normalize_obs(obs[batch_inds, 0], env))
            if self.optimize_memory_usage:
                for obs in self.observations:
                    next_observation_samples.append(
                        self._normalize_obs(obs[(batch_inds + 1) % self.buffer_size, 0], env))
            else:
                for next_obs in self.next_observations:
                    next_observation_samples.append(self._normalize_obs(next_obs[batch_inds, 0], env))
            observation_samples = tuple(observation_samples)
            next_observation_samples = tuple(next_observation_samples)
        else:
            observation_samples = self._normalize_obs(self.observations[batch_inds, 0, :], env)
            if self.optimize_memory_usage:
                next_observation_samples = self._normalize_obs(
                    self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
            else:
                next_observation_samples = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        observation_samples = self.augment_obs(observation_samples)
        next_observation_samples = self.augment_obs(next_observation_samples)

        data = (
            observation_samples,
            self.actions[batch_inds, 0, :],
            next_observation_samples,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
