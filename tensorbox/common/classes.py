import json
import os
from collections import namedtuple
from abc import ABC, abstractmethod
import tensorflow as tf

import tensorbox.common.utils as utils
import tensorbox.datasets.data_utils as du

Trajectory = namedtuple('Trajectory', ['observations',
                                       'actions',
                                       'rewards',
                                       'dones',
                                       'values',
                                       'mean_episode_return',
                                       'horizon'])


class BaseHook(ABC):
    """ Abstract Hook Class """

    @abstractmethod
    def final(self):
        """ final call of hook"""
        pass

    @abstractmethod
    def hook(self):
        pass


class Configuration(object):
    """ This class holds information about the settings of the current run."""
    def __init__(self,
                 net,
                 opt,
                 method,
                 dataset,
                 logger,
                 log_dir,
                 config_file_name='config.json',
                 **kwargs):
        self.net = net
        self.opt = opt
        self.method = method
        self.dataset = dataset
        self.logger = logger
        self.log_dir = log_dir
        self.kwargs = kwargs
        self.config_file_name = config_file_name

    def as_dict(self):
        """ Returns current configuration as dictionary."""
        return dict(method=self.method.get_config(),
                    network=self.net.get_config(),
                    dataset=self.dataset.get_config(),
                    optimizer=self.opt.get_config(),
                    log_dir=self.log_dir)

    def dump(self):
        """ Dumps the configuration to the disk at the specified log directory."""
        str_dict = utils.convert_to_string_only_dict(self.as_dict())

        file_name = os.path.join(self.log_dir, self.config_file_name)
        with open(file_name, 'w') as fp:
            json.dump(str_dict, fp, sort_keys=True, indent=4, separators=(',', ': '))
        print('Created log file: {}'.format(file_name))


class DatasetWrapper(object):
    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 name,
                 batch_size=32,
                 wrapped_class=None):

        assert isinstance(batch_size, int), 'DatasetWrapper: batch size must be an integer value'
        assert isinstance(name, str), 'DatasetWrapper: name must be of type str'
        # tf.Dataset() placeholders
        self.train = None
        self.test = None
        self.val = None
        self.name = name

        # Numpy placeholders
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.x_shape = x_train.shape[1:]
        self.y_shape = y_train.shape[1:]
        self.x_mean = None
        self.y_mean = None
        self.x_std = None
        self.y_std = None
        self.normalized = False

        # define pre-sets for child classes
        self.wrapped_class = wrapped_class
        self.plot_function = None
        self.batch_size = batch_size

    def build_tf_dataset(self, mappings=(), shuffle=True):
        """
        build tf.data.Dataset() from numpy arrays
        :param mappings: function, call these functions and apply them to the tf.Dataset()
        :param shuffle: bool,
        :return:
        """
        assert isinstance(mappings, tuple), 'mappings must be of type tuple (immutable)'
        self.train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        if self.x_test and self.y_test:  # only if test set is provided
            self.test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        for func in mappings:
            self.train = self.train.map(func)
            if self.x_test and self.y_test:
                self.test = self.test.map(func)

        if shuffle:
            self.train = self.train.shuffle(self.batch_size * 64)
        buffer_size = 16
        self.train = self.train.batch(self.batch_size).prefetch(buffer_size)
        if self.x_test and self.y_test:
            self.test = self.test.batch(self.batch_size).prefetch(buffer_size)

    def get_config(self):
        return dict(name=self.name,
                    size=len(self.x_train),
                    x_shape=self.x_shape,
                    y_shape=self.y_shape,
                    batch_size=self.batch_size)

    def normalize_data(self):
        """
        normalize datasets according to mean, stddev
        :return: None
        """
        self.x_mean, self.x_std = du.get_mean_std(self.x_train)
        self.y_mean, self.y_std = du.get_mean_std(self.y_train)
        self.x_train = du.normalize(self.x_train, self.x_mean, self.x_std)
        self.y_train = du.normalize(self.y_train, self.y_mean, self.y_std)
        if self.x_test and self.y_test:
            self.x_test = du.normalize(self.x_test, self.x_mean, self.x_std)
            self.y_test = du.normalize(self.y_test, self.y_mean, self.y_std)
        self.normalized = True

    def plot(self):
        assert self.wrapped_class, 'Plot()-function is not defined for this dataset!'
        self.wrapped_class.plot_data_set(self.x_train, self.y_train,
                                         window_title_suffix='Train Set (normalized)')

    def plot_predictions(self, y_pred):
        assert self.wrapped_class, 'Plot()-function is not defined for this dataset!'
        self.wrapped_class.plot_data_set(self.x_test, y_pred,
                                             window_title_suffix='Predictions (normalized)')

    def set_wrapped_function(self, func):
        self.wrapped_class = func

