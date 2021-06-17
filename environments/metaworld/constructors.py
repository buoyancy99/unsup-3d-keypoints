import numpy as np
import mujoco_py
import gym
import metaworld
from metaworld.envs.asset_path_utils import full_v1_path_for
from gym.utils import seeding
import os

from metaworld.envs.mujoco.sawyer_xyz.v1 import (
    SawyerBoxCloseEnv,
    SawyerDoorEnv,
    SawyerDrawerCloseEnv,
    SawyerHammerEnv,
    SawyerPegUnplugSideEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerWindowOpenEnv,
)


class MujocoMulticamMixin(object):
    def __init__(self, mode):
        super().__init__()
        if mode == 'train':
            self.tasks = metaworld.ML1(self.task_name).train_tasks
        elif mode == 'test':
            self.tasks = metaworld.ML1(self.task_name).test_tasks

        self.info_keywords = ('success', )
        self.camera_names = ['cam1', 'cam2', 'cam3']
        self.camera_width = 128
        self.camera_height = 128
        self.fov = 80
        self.view_matrices = None
        self.projection_matrix = None
        self.seed()
        self._compute_camera_matrices()

    def get_first_frame(self):
        xml_path = full_v1_path_for('sawyer_xyz/sawyer_empty.xml')
        model = mujoco_py.load_model_from_path(xml_path)
        sim = mujoco_py.MjSim(model)
        images = self._get_camera_images(sim)
        del sim
        del model
        return images

    @property
    def observation_space(self):
        if self.obs_type == 'hybrid':
            # robot_joints here is the gripper opening-closing state as it's hardly visible from a distance
            # robot_joint_positions is just place holder, containing all zeroes 
            observation_space = gym.spaces.Dict({
                'images': gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.camera_names),
                                                                   self.camera_height,
                                                                   self.camera_width,
                                                                   3), dtype=np.uint8),
                'robot_joints': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                'robot_joint_positions': gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 3))
            })

            return observation_space
        elif self.obs_type == 'state':
            return super().observation_space

    def connect_gui(self, rank):
        pass

    def _compute_camera_matrices(self):
        view_matrices = []
        for camera_name in self.camera_names:
            rot_mat = self.sim.data.get_camera_xmat(camera_name)
            pos = self.sim.data.get_camera_xpos(camera_name)
            view_matrix = np.eye(4)
            view_matrix[:3, :3] = rot_mat.T
            view_matrix[:3, 3] = - rot_mat.T @ pos
            mean_pos = np.array([0.02, 0.66, 0.08])
            normalization_matrix = np.diag([0.15, 0.18, 0.3, 1])
            normalization_matrix[:3, 3] = mean_pos

            """
            Notice normalization matrix is multiplied into the view matrix for this environments. 
            However, when decoding, normalization and unnormalization cancels out, so no need to worry
            about decoding. Only the latent is affected (correctly normalized).
            """

            view_matrix = view_matrix @ normalization_matrix
            view_matrices.append(view_matrix)

        self.view_matrices = np.stack(view_matrices, 0).astype(np.float32)

        projection_matrix = np.zeros((4, 4))
        # fovy = self.sim.model.cam_fovy[cam_id]
        fov = self.fov
        scale = 1
        near = 0.01
        far = 50
        projection_matrix[0, 0] = 1 / np.tan(np.deg2rad(fov) / 2)
        projection_matrix[1, 1] = 1 / np.tan(np.deg2rad(fov * scale) / 2)
        projection_matrix[2, 2] = (near + far) / (near - far)
        projection_matrix[2, 3] = 2 * near * far / (near - far)
        projection_matrix[3, 2] = -1

        self.projection_matrix = projection_matrix.astype(np.float32)

    def _get_camera_images(self, sim):
        images = []

        for camera_name in self.camera_names:
            camera_obs = sim.render(
                camera_name=camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=False,
            )

            images.append(np.flipud(camera_obs))
            # images.append(camera_obs)

        images = np.stack(images, 0)

        return images

    def _get_hybrid_obs(self):
        if self.obs_type == 'state':
            state = super()._get_obs()
            return state
        elif self.obs_type == 'hybrid':
            images = self._get_camera_images(self.sim)
            # robot_joints = self.get_endeff_pos()
            robot_joints = np.array(self.sim.data.ctrl[0]).reshape((1, ))
            robot_joint_positions = np.zeros(robot_joints.shape + (3, ))
            obs = dict(images=images,
                       robot_joints=robot_joints,   # not actually joints, but only gripper state as it's hardly visible from a distance
                       robot_joint_positions=robot_joint_positions)  # not actually joint positions, all 0
            return obs

    def reset(self):
        num_tasks = len(self.tasks)
        task_idx = self.np_random.randint(num_tasks)
        self.set_task(self.tasks[task_idx])
        _ = super().reset()
        obs = self._get_hybrid_obs()

        return obs

    def step(self, action):
        _, reward, done, info = super().step(action)
        obs = self._get_hybrid_obs()

        return obs, reward, done, info


class HybridHammerEnvV1(MujocoMulticamMixin, SawyerHammerEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'hammer-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridDrawerCloseEnvV1(MujocoMulticamMixin, SawyerDrawerCloseEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'drawer-close-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridReachPushWallEnvV1(MujocoMulticamMixin, SawyerReachPushPickPlaceWallEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'push-wall-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridBoxCloseEnvV1(MujocoMulticamMixin, SawyerBoxCloseEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'box-close-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridDoorOpenEnvV1(MujocoMulticamMixin, SawyerDoorEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'door-open-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridPegUnplugSideEnvV1(MujocoMulticamMixin, SawyerPegUnplugSideEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'peg-unplug-side-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridWindowOpenEnvV1(MujocoMulticamMixin, SawyerWindowOpenEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'window-open-v1'
        MujocoMulticamMixin.__init__(self, mode)


class HybridPickPlaceEnvV1(MujocoMulticamMixin, SawyerReachPushPickPlaceEnv):
    def __init__(self, mode):
        self.obs_type = 'hybrid'
        self.task_name = 'pick-place-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateHammerEnvV1(MujocoMulticamMixin, SawyerHammerEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'hammer-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateDrawerCloseEnvV1(MujocoMulticamMixin, SawyerDrawerCloseEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'drawer-close-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateReachPushWallEnvV1(MujocoMulticamMixin, SawyerReachPushPickPlaceWallEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'push-wall-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateBoxCloseEnvV1(MujocoMulticamMixin, SawyerBoxCloseEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'box-close-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateDoorOpenEnvV1(MujocoMulticamMixin, SawyerDoorEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'door-open-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StatePegUnplugSideEnvV1(MujocoMulticamMixin, SawyerPegUnplugSideEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'peg-unplug-side-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StateWindowOpenEnvV1(MujocoMulticamMixin, SawyerWindowOpenEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'window-open-v1'
        MujocoMulticamMixin.__init__(self, mode)


class StatePickPlaceEnvV1(MujocoMulticamMixin, SawyerReachPushPickPlaceEnv):
    def __init__(self, mode):
        self.obs_type = 'state'
        self.task_name = 'pick-place-v1'
        MujocoMulticamMixin.__init__(self, mode)
