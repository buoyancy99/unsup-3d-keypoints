from algorithms.ppo_keypoint.ppo_keypoint_algo import KeypointPpoAlgo
from algorithms.ppo_keypoint.ppo_keypoint_policy import MultiviewKeypointPpoPolicy
from algorithms.common.extractors import KeypointHybridExtractor
from algorithms.common.models.cnn import cnn_registry
from algorithms.common.models.keypoint_net import KeypointNet3d
from experiments.common.base_config import ppo_pixel_config_registry
from experiments.common.experiment import KeypointExperiment


class PpoKeypointExperiment(KeypointExperiment):
    def __init__(self, args):
        assert args.mode is not None, 'Must specify mode for keypoint net [2d, 3d]'
        super(PpoKeypointExperiment, self).__init__(
            KeypointPpoAlgo,
            MultiviewKeypointPpoPolicy,
            args,
            ppo_pixel_config_registry,
            'PPO_Keypoint{}{}'.format(args.mode, 'Ours' if args.augment and args.mode == '2d' else ''),
        )

        env_metadata = self._get_env_metadata()

        self.algo_kwargs.update(
            policy_kwargs=dict(
                features_extractor_class=KeypointHybridExtractor,
                features_extractor_kwargs=dict(
                    num_keypoints=args.num_keypoints,
                    keypoint_dim=2 if args.mode == '2d' else 3,
                    renormalize=True,
                    state_feature_keys=['robot_joints'] if args.use_hybrid_feature else []
                ),
                augment=args.augment if args.mode == '2d' else True,
                unsup_net_class=KeypointNet3d,
                unsup_net_kwargs=dict(
                    projection_matrix=env_metadata['projection_matrix'],
                    view_matrices=env_metadata['view_matrices'],
                    num_keypoints=args.num_keypoints,
                    encoder_cls=cnn_registry[args.cnn_type][0],
                    decoder_cls=cnn_registry[args.cnn_type][1],
                    n_filters=args.n_filters,
                    separation_margin=args.separation_margin,
                    mean_depth=args.mean_depth,
                    noise=args.noise,
                    independent_pred=args.independent_pred,
                    decode_first_frame=args.decode_first_frame,
                    decode_attention=args.decode_attention
                ),
                train_jointly=args.train_jointly,
                latent_stack=args.latent_stack,
                offset_crop=args.offset_crop,
                first_frame=env_metadata['first_frame'] if args.decode_first_frame else None
            ),
            unsup_coef_dict=dict(
                ae_loss=args.ae_coef,
                sparsity_loss=args.sparsity_coef,
                multiview_loss=args.multiview_coef,
                separation_loss=args.separation_coef
            ),
            buffer_size=args.buffer_size,
            unsup_steps=args.unsup_steps,
            unsup_gamma=args.unsup_gamma
        )


if __name__ == '__main__':
    from experiments.common.parse import get_args
    args = get_args()
    experiment = PpoKeypointExperiment(args)
    experiment.train()

