import numpy as np
import pathlib
import gym
import os
import cv2
import torch
import pickle
import random
from tqdm import tqdm

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import environments
from algorithms.common.callbacks import VisualizeCallback
from environments.wrappers.frame_stack import FrameStackWrapper
from environments.wrappers.multitask_norm import MultiTaskNormalize
from stable_baselines3.sac.sac import SAC
from stable_baselines3.ppo.ppo import PPO


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writter = cv2.VideoWriter('images/output.avi', fourcc, 20.0, (1024, 1024))


class BaseExperiment:
    def __init__(
        self,
        algo_class,
        algo_kwargs,
        policy_class,
        env_id,
        exp_name,
        total_timesteps,
        num_envs,
        save_freq,
        seed
    ):
        self.algo_class = algo_class
        self.algo_kwargs = algo_kwargs
        self.policy_class = policy_class
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.exp_name = exp_name
        self.save_freq = save_freq // num_envs if issubclass(self.algo_class, PPO) else save_freq
        self.seed = seed

        self.continue_from = 0
        self.buffer = None

        self.model_dir = os.path.join('ckpts', self.exp_name)
        self.log_dir = os.path.join('data', self.exp_name)
        self.buffer_dir = os.path.join('buffer', self.exp_name)
        self.image_dir = os.path.join('images', self.exp_name)

        self.algo_kwargs.update(
            seed=seed,
            verbose=1,
            tensorboard_log=self.log_dir
        )

        self._make_dir()
        self.set_global_seed(self.seed)

    def _make_dir(self):
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.buffer_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.image_dir).mkdir(parents=True, exist_ok=True)

    def set_global_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        if issubclass(self.algo_class, SAC):
            env = self._get_venv(num_envs=1, train=True)
        elif issubclass(self.algo_class, PPO):
            env = self._get_venv(train=True)
        else:
            raise NotImplementedError('Currently only SAC and PPO are supported')

        algo = self.algo_class(
            self.policy_class,
            env,
            **self.algo_kwargs
        )

        callbacks = CallbackList(self._get_callback_list())

        algo.learn(total_timesteps=self.total_timesteps,
                   callback=callbacks,
                   tb_log_name=self.exp_name,
                   reset_num_timesteps=False)
        algo.save(os.path.join(self.model_dir, "{}.zip".format(self.exp_name)))
        env.save(os.path.join(self.model_dir, "vecnorm.pkl"))

        return

    def _eval_loop(self, env, ckpt_path=None, episodes=100, callback=lambda *args: False):
        algo = self._get_trained_policy(ckpt_path)
        reward_stat = []
        step = 0

        for episode in range(episodes):
            obs = env.reset()
            total_reward = 0
            done = np.zeros(0)

            while not done.any():
                action, _ = algo.predict(obs)
                stop = callback(step, algo, obs, action)
                if stop:
                    return

                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
            assert done.all()
            reward_stat.extend(total_reward.tolist())

        return reward_stat

    def eval(self):
        env = self._get_venv(self.num_envs, train=False)
        reward_stat = self._eval_loop(env, episodes=100)
        print('eval reward: {:.2f}', np.mean(reward_stat))

    def eval_all_ckpts(self):
        reward_stats = []
        step_stats = []
        ckpt_paths = self._get_ckpt_paths()
        env = self._get_venv(self.num_envs, train=False)

        for ckpt_path in tqdm(ckpt_paths):
            reward_stat = self._eval_loop(env, ckpt_path, episodes=5)
            reward_stats.append(reward_stat)
            step_stats.append(int(ckpt_path.split('_')[-2]))
        with open(os.path.join(self.log_dir, 'eval_stats.pkl'), 'wb') as file_handler:
            pickle.dump(dict(step_stats=step_stats, reward_stats=reward_stats), file_handler)

    def visualize(self):
        env = self._get_venv(1, train=False)
        self._eval_loop(env, ckpt_path=None, episodes=100, callback=self._visualize_obs)
        video_writter.release()

    def collect_rollouts(self):
        env = self._get_venv(self.num_envs, train=False)
        self._eval_loop(env, ckpt_path=None, episodes=128, callback=self._add_to_buffer)

    def _get_callback_list(self):
        checkpoint_callback = CheckpointCallback(save_freq=self.save_freq,
                                                 save_path=self.model_dir,
                                                 name_prefix=self.exp_name)

        callback_list = [checkpoint_callback]

        return callback_list

    def _setup_func(self, env, rank):
        return env

    def _get_env(self, rank=0, train=True):
        env = gym.make(self.env_id)
        env = self._setup_func(env, rank)

        if train:
            info_keywords = env.info_keywords if hasattr(env, 'info_keywords') else ()
            env = Monitor(env,
                          os.path.join(self.log_dir, str(rank)),
                          allow_early_resets=True,
                          info_keywords=info_keywords,
                          continue_from=self.continue_from)

        env.seed(self.seed + rank)

        return env

    def _get_env_maker(self, rank, train=True):
        return lambda: self._get_env(rank, train)

    def _get_debug_env(self):
        return self._get_env()

    def _get_venv(self, num_envs=None, train=True):
        if num_envs is None:
            num_envs = self.num_envs
        env_makers = [self._get_env_maker(rank, train) for rank in range(num_envs)]

        if self.num_envs > 1:
            venv = SubprocVecEnv(env_makers)
        else:
            venv = DummyVecEnv(env_makers)

        if self.normalize_reward and train:
            if 'mt50' in self.env_id or 'ml45' in self.env_id:
                venv = MultiTaskNormalize(venv, norm_obs=False)
            else:
                venv = VecNormalize(venv, norm_obs=False)

        return venv

    def _get_env_metadata(self):
        return {}

    def _get_ckpt_paths(self):
        ckpt_names = os.listdir(self.model_dir)
        ckpt_names = [name for name in ckpt_names if '.zip' in name and name != self.exp_name + '.zip']
        ckpt_names = sorted(ckpt_names, key=lambda filename: int(filename.split('_')[-2]))
        ckpt_paths = [os.path.join(self.model_dir, ckpt_name) for ckpt_name in ckpt_names]
        return ckpt_paths

    def _get_trained_policy(self, ckpt_path):
        if ckpt_path is None:
            ckpt_path = self._get_ckpt_paths()[-1]
        algo = self.algo_class.load(ckpt_path)
        return algo

    def _visualize_obs(self, step, algo, obs, action):
        return False

    def _add_to_buffer(self, step, algo, obs, action):
        if self.buffer is None:
            self.buffer = dict(
                obs={k: np.empty((0,) + v.shape, dtype=v.dtype) for k, v, in algo.observation_space.spaces.items()},
                action=np.empty((0,) + algo.action_space.shape, dtype=algo.action_space.dtype)
            )

        for k in self.buffer['obs'].keys():
            self.buffer['obs'][k] = np.append(self.buffer['obs'][k], obs[k], 0)
        self.buffer['action'] = np.append(self.buffer['action'], action, 0)

        if len(self.buffer['action']) == 4096:
            with open(os.path.join(self.buffer_dir, 'eval_buffer_{}.pkl'.format(step + 1)), 'wb') as file_handler:
                pickle.dump(self.buffer, file_handler, protocol=4)
            self.buffer = None



class ProjExperiment(BaseExperiment):
    def __init__(self, algo_class, policy_class, args, config_registry, exp_name):
        base_config = config_registry[args.task]()
        for k, _ in base_config.items():
            if k in vars(args) and getattr(args, k) is not None:
                base_config[k] = getattr(args, k)

        env_id = '{}{}Env-v{}'.format('Hybrid', args.task.capitalize(), args.env_version)
        exp_name = '_'.join([str(args.exp_id), exp_name, env_id])
        self.frame_stack = args.frame_stack
        self.normalize_reward = True if issubclass(algo_class, PPO) else False

        # if args.offset_crop:
        #     assert args.augment, 'offset_crop only when using augmentation'

        super(ProjExperiment, self).__init__(
            algo_class,
            base_config,
            policy_class,
            env_id,
            exp_name,
            args.total_timesteps,
            args.num_envs,
            args.save_freq,
            args.seed,
        )

    def _setup_func(self, env, rank):
        env = FrameStackWrapper(env, self.frame_stack)
        env.connect_gui(rank=rank)
        return env

    def _visualize_obs(self, step, algo, obs, action):
        images = obs['images'][0]
        for cam_id in range(images.shape[1]):
            cv2.imshow('cam{}'.format(cam_id),
                       cv2.resize(cv2.cvtColor(images[0][cam_id][..., :3], cv2.COLOR_RGB2BGR), (512, 512),
                                  interpolation=cv2.INTER_NEAREST))

        k = cv2.waitKey(5)
        stop = k == ord('q')

        return stop


class KeypointExperiment(ProjExperiment):
    def _get_callback_list(self):
        callback_list = super(ProjExperiment, self)._get_callback_list()

        visualize_callback = VisualizeCallback(save_freq=self.save_freq // 4,
                                               save_path=self.image_dir,
                                               name_prefix=self.exp_name)
        callback_list.append(visualize_callback)

        return callback_list

    def _get_env_metadata(self):
        env = self._get_env(train=False)
        projection_matrix = np.array(env.projection_matrix)
        view_matrices = np.array(env.view_matrices)
        first_frame = env.get_first_frame()
        env.close()
        del env

        metadata = dict(
            projection_matrix=projection_matrix,
            view_matrices=view_matrices,
            first_frame=first_frame
        )
        return metadata

    def _visualize_obs(self, step, algo, obs, action):
        obs = {k: torch.from_numpy(v).to(algo.policy.device) for k, v in obs.items()}
        obs = algo.policy.process_uncropped_obs(obs)

        vis_image = algo.policy.visualize(obs)[0].permute(1, 2, 0)
        vis_image = (vis_image.detach().cpu().numpy() * 255.0).astype(np.uint8)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        h = vis_image.shape[0]

        cv2.imshow('keypoints', vis_image[:, h:])
        recon = cv2.resize(vis_image[:, :h], (1024, 1024))
        cv2.imshow('recon', recon)
        video_writter.write(recon)

        k = cv2.waitKey(5)
        stop = k == ord('q')

        # save keypoint
        if k == ord('s'):
            cv2.imwrite(os.path.join(self.image_dir, '{}.png'.format(step)), vis_image[:, h//3:2*h//3])

        return stop
