import argparse

parser = argparse.ArgumentParser(description='IMPALA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu number')

parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-grad-norm', type=float, default=50)
parser.add_argument('--value-loss-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--n-step', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--coef-hat', type=float, default=1.0)
parser.add_argument('--rho-hat', type=float, default=1.0)

# 游戏配置
parser.add_argument('--env', type=str, default='Phoenix-Atari2600')
parser.add_argument('--s-channel', type=int, default=3)
parser.add_argument('--a-space', type=int, default=8)
parser.add_argument('--max-episode-length', type=int, default=100000)


def get_opt(args=[]):
    opt = parser.parse_args(args)
    return opt
