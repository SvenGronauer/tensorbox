import os
import numpy as np
import argparse
import datetime
import sys
import tensorflow as tf
import gym
import roboschool
from gym import envs


def get_default_args(func_name='testing', debug_level=0):
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='This is the default parser.')
    parser.add_argument('--alg', default=os.cpu_count(), type=int,
                        help='Number of available CPU cores.')
    parser.add_argument('--cores', default=os.cpu_count(), type=int,
                        help='Number of available CPU cores.')
    parser.add_argument('--debug', default=debug_level, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--env', default='CartPole-v0', type=str,
                        help='Default environment for RL algorithms')
    parser.add_argument('--func', dest='func', default=func_name,
                        help='Specify function name to be testing')
    parser.add_argument('--log', dest='log_path', default='/var/tmp/ga87zej/',
                        help='Set the seed for random generator')
    args = parser.parse_args()
    if not args.log_path:
        default_path = '/var/tmp/ga87zej/'
        args.log_path = os.path.join(default_path, args.func,
                                     datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    return args


def normalize(xs,
              axis=None,
              eps=1e-8):
    """ Normalize array along axis
    :param xs: np.array(), array to normalize
    :param axis: int, axis along which is normalized
    :param eps: float, offset to avoid division by zero
    :return: np.array(), normed array
    """
    return (xs - xs.mean(axis=axis)) / (xs.std(axis=axis) + eps)


def make_env(env_name, seed=None):
    """ Creates a Gym Environment
    :param env_name: str, name of environment
    :param seed: int, make experiments deterministic
    :return:
    """
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed=seed)
    all_gym_environments = [env_spec.id for env_spec in envs.registry.all()]

    if env_name in all_gym_environments:
        return gym.make(env_name)
    else:
        raise ValueError('Did not find environment with name: {}'.format(env_name))


def mkdir(path):
    """ create directory at given path
    :param path: str, path
    :return: bool, True if created directories
    """
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # os.mkdir(path)
        created_dir = True
    return created_dir


def safe_mean(xs):
    """ Calculate mean value of an array safely and avoid division errors
    :param xs: np.array, array to calculate mean
    :return: np.float, mean value of xs
    """
    return np.nan if len(xs) == 0 else np.mean(xs)
