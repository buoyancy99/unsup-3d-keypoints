from environments.locomotion.gym_locomotion_envs import AntBulletEnv, AntNocolorBulletEnv


class StateAntEnvV1(AntBulletEnv):
    def __init__(self):
        super().__init__(obs_type='state')


class HybridAntEnvV1(AntBulletEnv):
    def __init__(self):
        super().__init__(obs_type='hybrid')


class HybridAntncEnvV1(AntNocolorBulletEnv):
    def __init__(self):
        super().__init__(obs_type='hybrid')

