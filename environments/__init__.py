from gym.envs.registration import register

register(
    id='HybridHammerEnv-v1',
    entry_point='environments.metaworld.constructors:HybridHammerEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridDcEnv-v1',
    entry_point='environments.metaworld.constructors:HybridDrawerCloseEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridPwEnv-v1',
    entry_point='environments.metaworld.constructors:HybridReachPushWallEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridBcEnv-v1',
    entry_point='environments.metaworld.constructors:HybridBoxCloseEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridDoorEnv-v1',
    entry_point='environments.metaworld.constructors:HybridDoorOpenEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridPusEnv-v1',
    entry_point='environments.metaworld.constructors:HybridPegUnplugSideEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridWoEnv-v1',
    entry_point='environments.metaworld.constructors:HybridWindowOpenEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridPickplaceEnv-v1',
    entry_point='environments.metaworld.constructors:HybridPickPlaceEnvV1',
    max_episode_steps=200,
    kwargs={'mode': 'train'}
)

register(
    id='HybridAntEnv-v1',
    entry_point='environments.locomotion.constructors:HybridAntEnvV1',
    max_episode_steps=1000,
    reward_threshold=2500.0
)

register(
    id='HybridAntncEnv-v1',
    entry_point='environments.locomotion.constructors:HybridAntncEnvV1',
    max_episode_steps=1000,
    reward_threshold=12500.0
)

