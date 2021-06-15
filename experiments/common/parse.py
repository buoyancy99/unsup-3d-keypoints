import argparse


def get_args():
    parser = argparse.ArgumentParser(description='parameters for training and visualization')
    parser.add_argument('--algo', '-a', type=str, help='algorithm',
                        choices=['ppopixel', 'pporad', 'ppokeypoint'])
    parser.add_argument('--task', '-t', type=str, help='task to perform', required=True)
    parser.add_argument('--obs', choices=['hybrid', 'state'], default='hybrid', help='observation type')
    parser.add_argument('--env_version', '-v', type=int, help='version of environment', required=True)
    parser.add_argument('--exp_id', '-e', type=str, help='experiment id', required=True)
    parser.add_argument('--exp_name', type=str, default='', help='experiment id')
    parser.add_argument('--augment', action='store_true', help='enable random crop augmentation')
    parser.add_argument('--frame_stack', type=int, default=1, help='how many frames to stack')
    parser.add_argument('--use_hybrid_feature', '-u', action='store_true', help='Include joint states to policy')
    parser.add_argument('--latent_stack', '-l', action='store_true', help='share encoder for each frame in frame stack')
    parser.add_argument('--independent_pred', '-i', action='store_true', help='prediction from each view independently')
    parser.add_argument('--n_filters', '-f', type=int, default=16, help='Number of base filters for cnn encoder')
    parser.add_argument('--cnn_type', '-c', type=str, default='custome', choices=['nature', 'custome'], help='CNN type')
    parser.add_argument('--mean_depth', type=float, default=0.5, help='mean_depth for camera to target in keypoint3d')
    parser.add_argument('--decode_first_frame', '-d', action='store_true', help='decode first frame in keypoint net')
    parser.add_argument('--decode_attention', action='store_true', help='multiply attention to gaussian map in keypoint net')
    parser.add_argument('--offset_crop', '-o', action='store_true', help='offset crop in keypoint net')

    parser.add_argument('--num_envs', type=int, default=16, help='number of parallel environments for ppo')
    parser.add_argument('--total_timesteps', type=int, default=int(2e6), help='Number of total env steps to train')
    parser.add_argument('--save_freq', type=int, default=1000000, help='Save frequency in env steps')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed for environments')

    # PPO Common
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help='obs/replay buffer size')
    parser.add_argument('--n_steps', type=int, help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', type=int, help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, help='Number of steps for each environment per update ')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, help='Factor for trade-off  for Generalized Advantage Estimator')
    parser.add_argument('--clip_range', type=float, help='ppo clip parameter')
    parser.add_argument('--clip_range_vf', type=float, help='ppo clip parameter')
    parser.add_argument('--ent_coef', type=float, help='entropy term coefficient')
    parser.add_argument('--vf_coef', type=float, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, help='The maximum value for the gradient clipping ')
    parser.add_argument('--target_kl', type=float, help='The maximum value for the gradient clipping ')

    # Unsup Common
    parser.add_argument('--train_jointly', '-j', action='store_true', help='Train ae with policy jointly')

    # Keypoint Common
    parser.add_argument('--mode', '-m', choices=['2d', '3d'], help='For keypoint methods only [2d, 3d]')
    parser.add_argument('--num_keypoints', type=int, default=32, help='Number of keypoints')
    parser.add_argument('--separation_margin', default=0.0, type=float, help='Margin for separation loss')
    parser.add_argument('--noise', '-n', type=float, default=0.05, help='Maximum noise to keypoints before decoding')
    parser.add_argument('--unsup_steps', type=int, default=400, help='number of ae optimization steps each update loop')
    parser.add_argument('--unsup_gamma', type=float, default=1.0, help='ae step exp decline coefficient')
    parser.add_argument('--ae_coef', type=float, default=5.0, help='autoencode term coefficient')
    parser.add_argument('--sparsity_coef', type=float, default=0.005, help='sparse keypoint loss coefficient, use hp in Honglak Lees paper')
    parser.add_argument('--multiview_coef', type=float, default=0.05, help='multiview consistancy loss coefficient')
    parser.add_argument('--separation_coef', type=float, default=0.0025, help='separation loss coefficient')

    args = parser.parse_args()

    return args
