import gym
import numpy as np
import pybullet as p
import os

from pybullet_utils import bullet_client

from pkg_resources import parse_version

os.environ["PYBULLET_EGL"] = "1"

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

from environments.locomotion.config import global_config, empty_config


class MJCFBaseBulletEnv(gym.Env):
    """
      Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
      These environments create single-player scenes and behave like normal Gym environments, if
      you don't use multiplayer.
      """

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, robot, obs_type, gui=False, config=empty_config):
        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.camera = Camera()
        self.isRender = gui
        self.isDebug = False
        self.robot = robot
        self.obs_type = obs_type
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240

        self.action_space = robot.action_space

        # self.reset()
        self.config = None
        self._setup_config(config)
        self.projection_matrix, self.view_matrices = self._compute_camera_matrices()

        if obs_type == 'state':
            self.observation_space = robot.observation_space
        else:
            self.observation_space = gym.spaces.Dict({
                'images': gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.obs_camera_config['eye_positions']),
                                                                   self.obs_camera_config['image_obs_resolution'],
                                                                   self.obs_camera_config['image_obs_resolution'],
                                                                   3), dtype=np.uint8),
                'robot_joints': gym.spaces.Box(low=-5.0, high=5.0, shape=(2, )),
                'robot_joint_positions': gym.spaces.Box(low=0.0, high=1.0, shape=(2, 3))
            })

    def _setup_config(self, config):
        config = global_config + config
        self.obs_camera_config = config.obs_camera_config
        self.config = config

    def get_first_frame(self):
        self.reset()
        return self._get_image()

    def _compute_camera_matrices(self, base_pos=[0, 0, 0], normalize=True):
        width, height = self.obs_camera_config.image_obs_resolution, self.obs_camera_config.image_obs_resolution

        base_pos = np.array(base_pos)
        eyes = (np.array(self.obs_camera_config.eye_positions) + base_pos[None]).tolist()
        targets = (np.array(self.obs_camera_config.eye_targets) + base_pos[None]).tolist()
        ups = self.obs_camera_config.eye_ups
        fov = self.obs_camera_config.fov
        aspect = width / height
        near, far = self.obs_camera_config.near, self.obs_camera_config.far
        projection_matrix = np.array(p.computeProjectionMatrixFOV(fov, aspect, near, far,
                                                                  physicsClientId=self.physicsClientId))
        projection_matrix = projection_matrix.reshape(4, 4).T
        view_matrices = []
        for eye, target, up in zip(eyes, targets, ups):
            view_matrix = np.array(p.computeViewMatrix(eye, target, up, physicsClientId=self.physicsClientId))
            view_matrix = view_matrix.reshape(4, 4).T

            if normalize:
                """
                Notice normalization matrix is multiplied into the view matrix for this environments. 
                However, when decoding, normalization and unnormalization cancels out, so no need to worry
                about decoding. Only the latent is affected (correctly normalized).
                """
                mean_pos = np.array([0.02, -0.05, 0.25])
                normalization_matrix = np.diag([0.3, 0.34, 0.154, 1])
                normalization_matrix[:3, 3] = mean_pos
                view_matrix = view_matrix @ normalization_matrix
            view_matrices.append(view_matrix)
        view_matrices = np.array(view_matrices)

        return projection_matrix, view_matrices

    def _get_image(self):
        base_pos = [0, 0, 0]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = list(self.robot.body_xyz)

        base_pos[2] = 0  # stablize camera by making z shift 0

        projection_matrix, view_matrices = self._compute_camera_matrices(base_pos, normalize=False)
        width, height = self.obs_camera_config.image_obs_resolution, self.obs_camera_config.image_obs_resolution
        observation_images = []

        for view_matrix in view_matrices:
            (width, height, rgb_pixels,
             depth_pixels, segmentation_mask_buffer) = self._p.getCameraImage(width,
                                                                              height,
                                                                              view_matrix.T.flatten().tolist(),
                                                                              projection_matrix.T.flatten().tolist(),
                                                                              shadow=True,
                                                                              renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                                              physicsClientId=self.physicsClientId)

            rgb_pixels = np.array(rgb_pixels).reshape((width, height, -1))[:, :, :3].astype(np.uint8)
            depth_pixels = np.array(depth_pixels).reshape((width, height, -1))
            observation_images.append(rgb_pixels)

        rgb_pixels = np.stack(observation_images, 0)

        return rgb_pixels

    def connect_gui(self, rank):
        pass

    def enable_debug(self):
        # self.isRender = True
        self.isDebug = True


    def configure(self, args):
        self.robot.args = args

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def _get_obs(self):
        if self.obs_type == 'state':
            obs = self.robot.calc_state()
        else:
            x, y, z = self.robot.body_xyz
            robot_joints = np.array([np.cbrt(x) - 1, y])
            obs = {
                'images': self._get_image(),
                'robot_joints': robot_joints,
                'robot_joint_positions': np.zeros(self.observation_space['robot_joint_positions'].shape)  # NA now
            }
            # print(np.cbrt(np.array(self.robot.body_xyz))[:2])
        return obs

    def reset(self):
        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            # optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode == self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if egl:
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        self.robot.reset(self._p)
        for _ in range(1):
            self.scene.global_step()
        self.potential = self.robot.calc_potential()

        obs = self._get_obs()

        return obs

    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz
        if (self.physicsClientId >= 0):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=self._cam_dist,
                                                                    yaw=self._cam_yaw,
                                                                    pitch=self._cam_pitch,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                                      height=self._render_height,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            try:
                # Keep the previous orientation of the camera set by the user.
                con_mode = self._p.getConnectionInfo()['connectionMethod']
                if con_mode == self._p.SHARED_MEMORY or con_mode == self._p.GUI:
                    [yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
                    self._p.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            except:
                pass

        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1

    def HUD(self, state, a, done):
        pass

    # def step(self, *args, **kwargs):
    # 	if self.isRender:
    # 		base_pos=[0,0,0]
    # 		if (hasattr(self,'robot')):
    # 			if (hasattr(self.robot,'body_xyz')):
    # 				base_pos = self.robot.body_xyz
    # 				# Keep the previous orientation of the camera set by the user.
    # 				#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
    # 				self._p.resetDebugVisualizerCamera(3,0,0, base_pos)
    #
    #
    # 	return self.step(*args, **kwargs)
    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed


class Camera:

    def __init__(self):
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 10
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
