import os
import numpy as np
import argparse
import datetime
import sys


def get_default_args(func_name='testing', debug_level=0):
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='Test MNIST')
    parser.add_argument('--debug', default=debug_level, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--func', dest='func', default=func_name,
                        help='Specify function name to be testing')
    parser.add_argument('--log', dest='log_path', default='/var/tmp/ga87zej/',
                        help='Set the seed for random generator')
    args = parser.parse_args()
    args.log_path = os.path.join(args.log_path, args.func, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    return args


def mkdir(path):
    """ create directory at given path"""
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # os.mkdir(path)
        created_dir = True
    return created_dir


def safe_mean(xs):
    """
    Calculate mean value of an array safely and avoid division errors

    :param xs: np.array, array to calculate mean
    :return: np.float, mean value of xs
    """
    return np.nan if len(xs) == 0 else np.mean(xs)