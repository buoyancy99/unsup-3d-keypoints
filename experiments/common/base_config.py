"""
PPO Configs
"""

def get_ppo_base_config():
    config = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None
    )
    return config


def get_ppo_pixel_base_config():
    config = get_ppo_base_config()
    config.update(
        n_steps=256,
        max_grad_norm=2.0,
    )
    return config


def get_ppo_pixel_scarf_config():
    config = get_ppo_pixel_base_config()
    config.update(
        batch_size=64,
        n_steps=1024
    )
    return config


def get_ppo_pixel_ant_config():
    config = get_ppo_pixel_base_config()
    config.update(
        n_steps=1000,
    )
    return config


def get_ppo_pixel_metaworld_config():
    config = get_ppo_pixel_base_config()
    config.update(
        n_steps=400,
        target_kl=0.12,
        n_epochs=8
    )
    return config


ppo_pixel_config_registry = dict(
    scarf=get_ppo_pixel_scarf_config,
    ant=get_ppo_pixel_ant_config,
    antnc=get_ppo_pixel_ant_config,
    hammer=get_ppo_pixel_metaworld_config,
    dc=get_ppo_pixel_metaworld_config,
    pw=get_ppo_pixel_metaworld_config,
    bc=get_ppo_pixel_metaworld_config,
    door=get_ppo_pixel_metaworld_config,
    pus=get_ppo_pixel_metaworld_config,
    wo=get_ppo_pixel_metaworld_config,
    pickplace=get_ppo_pixel_metaworld_config,
    ml45=get_ppo_pixel_metaworld_config,
)
