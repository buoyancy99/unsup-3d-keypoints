from stable_baselines3.ppo.ppo import PPO
from algorithms.common.extractors import AugmentHybridCnnExtractor
from algorithms.common.models.cnn import cnn_registry
from algorithms.ppo_pixel.ppo_pixel_policy import MultiviewAugmentPpoPolicy
from experiments.common.base_config import ppo_pixel_config_registry
from experiments.common.experiment import ProjExperiment


class PpoPixelExperiment(ProjExperiment):
    def __init__(self, args):
        super(PpoPixelExperiment, self).__init__(
            PPO,
            MultiviewAugmentPpoPolicy,
            args,
            ppo_pixel_config_registry,
            'PPO_Pixel',
        )

        self.algo_kwargs.update(
            policy_kwargs=dict(
                features_extractor_class=AugmentHybridCnnExtractor,
                features_extractor_kwargs=dict(state_feature_keys=['robot_joints'] if args.use_hybrid_feature else [],
                                               encoder_kwargs=dict(n_filters=args.n_filters),
                                               encoder_cls=cnn_registry[args.cnn_type][0]
                                               ),
                augment=False
            )
        )


if __name__ == '__main__':
    from experiments.common.parse import get_args
    args = get_args()
    experiment = PpoPixelExperiment(args)
    experiment.train()

