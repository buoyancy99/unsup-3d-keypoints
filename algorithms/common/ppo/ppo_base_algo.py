import gym
import torch as th
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import trange
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import preprocess_obs

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.ppo.ppo import PPO

from ..buffers import AugmentRolloutBuffer, AugmentObsBuffer


class AugmentPpoAlgo(PPO):
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        self.rollout_buffer = AugmentRolloutBuffer(
            self.policy.crop_size,
            self.n_steps,
            self.observation_space,
            self.action_space,
            pass_offset=self.policy.offset_crop,
            first_frame=self.policy.first_frame,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)


class UnsupPpoAlgo(AugmentPpoAlgo):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[gym.Env, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        buffer_size: int = int(1e5),
        unsup_coef_dict: dict = {},
        unsup_steps: int = 1024,
        unsup_gamma: float = 1.0
    ):
        self.buffer_size = buffer_size
        self.unsup_coef_dict = {k.replace('coef', 'loss'): v for k, v in unsup_coef_dict.items()}
        self.unsup_steps = unsup_steps
        self.unsup_gamma = unsup_gamma

        super(UnsupPpoAlgo, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

    def _setup_model(self) -> None:
        super(UnsupPpoAlgo, self)._setup_model()

        crop_size = self.policy.crop_size if self.policy.augment else None
        self.obs_buffer = AugmentObsBuffer(
            crop_size,
            self.buffer_size,
            self.observation_space,
            self.action_space,
            pass_offset=self.policy.offset_crop,
            first_frame=self.policy.first_frame,
            device=self.device
        )

    def save_obs_buffer(self, path: str):
        """
        Save the replay buffer as a pickle file.
        :param path: (str) Path to the file where the replay buffer should be saved
        """
        assert self.obs_buffer is not None, "The replay buffer is not defined"
        with open(path, 'wb') as file_handler:
            pickle.dump(self.obs_buffer, file_handler, protocol=4)

    def load_obs_buffer(self, path: str):
        """
        Load a replay buffer from a pickle file.
        :param path: (str) Path to the pickled replay buffer.
        """
        with open(path, 'rb') as file_handler:
            self.obs_buffer = pickle.load(file_handler)
        assert isinstance(self.obs_buffer, AugmentObsBuffer), 'The obs buffer must inherit from DictObsBuffer class'

    def excluded_save_params(self):
        return super(UnsupPpoAlgo, self).excluded_save_params() + ["obs_buffer"]

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: (RolloutBuffer) Buffer to fill with rollouts
        :param n_steps: (int) Number of experiences to collect per environment
        :return: (bool) True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = rollout_buffer.to_torch(self._last_obs)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self.obs_buffer.add(self._last_obs)
            self._last_obs = new_obs
            self._last_dones = dones

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True

    def train_unsup(self):
        """
            :return loss, gradnorm: stats for logging
        """
        obs_data = self.obs_buffer.sample(self.batch_size, env=None)
        raise NotImplementedError

    def save_visualization(self, path, batch_size=8):
        raise NotImplementedError

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        grad_norms, unsup_grad_norms, policy_grad_norms = [], [], []
        unsup_losses = {k: [] for k in self.unsup_coef_dict.keys()}
        clip_fractions = []

        for _ in trange(self.unsup_steps, desc='Training Unsupervised Learning'):
            unsup_loss_dict, unsup_grad_norm = self.train_unsup()
            for k in unsup_losses.keys():
                if k in unsup_loss_dict:
                    unsup_losses[k].append(unsup_loss_dict[k].item())
            unsup_grad_norms.append(unsup_grad_norm)

        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # if self.num_timesteps >= 5000000:
                #     self.policy.features_extractor.eval()
                # else:
                #     self.policy.features_extractor.train()
                self.policy.features_extractor.train()

                values, log_prob, entropy, unsup_loss_dict = \
                    self.policy.evaluate_actions(rollout_data.observations, actions)

                # compute unsup loss
                unsup_loss_dict = {k: v.mean() for k, v in unsup_loss_dict.items()}
                unsup_loss = sum([v * self.unsup_coef_dict[k] for k, v in unsup_loss_dict.items()])
                for k in unsup_losses.keys():
                    if k in unsup_loss_dict:
                        unsup_losses[k].append(unsup_loss_dict[k].item())

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + unsup_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                joint_clip = max(self.max_grad_norm + 3.0, 15.0 - 0.01 * self._n_updates)
                policy_grad_norms.append(
                    th.nn.utils.clip_grad_norm_(self.policy.params_exclude_unsup, self.max_grad_norm).item())
                grad_norms.append(
                    th.nn.utils.clip_grad_norm_(self.policy.unsup_net.parameters(), joint_clip).item())

                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self.unsup_coef_dict = {k: v * self.unsup_gamma for k, v in self.unsup_coef_dict.items()}
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        for k, v in unsup_losses.items():
            logger.record("train/{}".format(k), np.mean(v))
        logger.record("train/grad_norm", np.mean(grad_norms))
        logger.record("train/unsup_grad_norm", np.mean(unsup_grad_norms))
        logger.record("train/policy_grad_norm", np.mean(policy_grad_norms))
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)
