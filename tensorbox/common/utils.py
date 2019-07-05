import argparse
import datetime
import os
import sys

import numpy as np
import tensorflow as tf


def convert_to_string_only_dict(input_dict):
    """
    Convert all values of a dictionary to string objects
    Useful, if you want to save a dictionary as .json file to the disk

    :param input_dict: dict, input to be converted
    :return: dict, converted string dictionary
    """
    converted_dict = dict()
    for key, value in input_dict.items():
        if isinstance(value, dict):  # transform dictionaries recursively
            converted_dict[key] = convert_to_string_only_dict(value)
        elif isinstance(value, type):
            converted_dict[key] = str(value.__name__)
        else:
            converted_dict[key] = str(value)
    return converted_dict


def get_default_args(func_name='testing',
                     log_dir='/var/tmp/ga87zej/',
                     debug_level=0):
    """ create the default arguments for program execution
    :param func_name:
    :param log_dir:
    :param debug_level: 
    :return: 
    """
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
    parser.add_argument('--log', dest='log_dir', default=log_dir,
                        help='Set the seed for random generator')

    args = parser.parse_args()
    args.log_dir = os.path.abspath(os.path.join(args.log_dir,
                                                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")))
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


def make_env(env_name, 
             seed=None):
    """ Creates a Gym Environment
    :param env_name: str, name of environment
    :param seed: int, make experiments deterministic
    :return:
    """
    import gym
    from gym import envs
    try:  # try to import Martin's environments
        from mygymenvs.environmentwrapper import get_data_wrapper
    except ImportError:
        print('WARNING: could not import mygymenvs! Hint: Did you install the progressbar package?')
    try:  # try to import roboschool environments
        import roboschool
    except ImportError:
        print('WARNING: could not import roboschool!')
    try:
        import pybullet_envs
        import pybulletgym.envs
    except ImportError:
        print('WARNING: could not import pybullet_envs!')

    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed=seed)
    all_gym_environments = [env_spec.id for env_spec in envs.registry.all()]

    if env_name in all_gym_environments:
        return gym.make(env_name)
    else:
        raise ValueError('Did not find environment with name: {}'.format(env_name))


def mkdir(path):
    """ create directory at a given path
    :param path: str, path
    :return: bool, True if created directories
    """
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        created_dir = True
    return created_dir


def safe_mean(xs):
    """ Calculate mean value of an array safely and avoid division errors
    :param xs: np.array, array to calculate mean
    :return: np.float, mean value of xs
    """
    return np.nan if len(xs) == 0 else float(np.mean(xs))
