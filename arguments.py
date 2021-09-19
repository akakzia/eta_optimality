import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--algo', default="SAC",
                        help='The rl algorithm to be used')
    parser.add_argument('--save-dir', default="./output",
                        help='Saving directory')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1e5), metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num-steps', type=int, default=1500001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')

    parser.add_argument('--gamma-one', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--gamma-two', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--update-frequency', type=int, default=1, metavar='N',
                        help='How many time to update critic 1 before 2')

    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--update-every', type=int, default=1, metavar='N',
                        help='after how many environment steps do the update')

    parser.add_argument('--max-episode-steps', type=int, default=1000, metavar='N',
                        help='Number of max environment steps per episode (default: 1000)')

    parser.add_argument('--start-steps', type=int, default=10000, metavar='N',
                        help='Number of episodes with random actions (default: 10000)')

    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay-size', type=int, default=100000, metavar='N',
                        help='size of replay buffer (default: 100000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')

    parser.add_argument('--eval-every', type=int, default=5, metavar='N',
                        help='Frequency of evaluations')
    parser.add_argument('--eval-episodes', type=int, default=2, metavar='N',
                        help='number of evaluation episodes')

    args = parser.parse_args()

    assert args.algo.lower() in ['sac', 'gsac']

    return args
