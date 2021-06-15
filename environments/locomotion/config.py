from attrdict import AttrDict


global_obs_camera_config = AttrDict(dict(
    image_obs_resolution=128,
    eye_positions=[(0.0, 0.0, 1.6), (-0.8, -0.6, 1.4), (0.8, -1.0, 0.5)],
    eye_targets=[(0.0, 0.0, 1.0), (0.0, 0.0, 0.4), (0.0, 0.0, 0.4)],
    eye_ups=[(0, 1, 0), (0, 0, 1), (0, 0, 1)],
    fov=80,
    near=0.001,
    far=10
))

global_config = AttrDict(dict(
    obs_camera_config=global_obs_camera_config,
))

empty_config = AttrDict()


ant_env_config = AttrDict(dict(
    obs_camera_config=dict(
        image_obs_resolution=128,
        eye_positions=[(0.0, 0.0, 1.6), (0.0, 1.8, 0.5), (1.8, 0.0, 0.5)],
        eye_targets=[(0.0, 0.0, 1.0), (0.0, 0.0, 0.3), (0.0, 0.0, 0.3)],
        eye_ups=[(0, -1, 0), (0, 0, 1), (0, 0, 1)],
        fov=80,
        near=0.001,
        far=10
    )
))


humanoid_env_config = AttrDict(dict(
    obs_camera_config=dict(
        image_obs_resolution=128,
        eye_positions=[(0.0, 0.0, 2.0), (0.0, 1.4, 0.8), (1.4, 0.0, 0.8)],
        eye_targets=[(0.0, 0.0, 1.0), (0.0, 0.0, 0.6), (0.0, 0.0, 0.6)],
        eye_ups=[(0, -1, 0), (0, 0, 1), (0, 0, 1)],
        fov=80,
        near=0.001,
        far=10
    )
))

