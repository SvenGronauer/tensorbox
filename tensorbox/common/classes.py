from collections import namedtuple
import tensorflow as tf
import tensorbox.datasets.data_utils as du


Trajectory = namedtuple('Trajectory', ['observations',
                                       'actions',
                                       'rewards',
                                       'dones',
                                       'values',
                                       'mean_episode_return',
                                       'horizon'])


class DatasetWrapper(object):
    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 batch_size=32,
                 wrapped_class=None):

        # tf.Dataset() placeholders
        self.train = None
        self.test = None
        self.val = None

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
        self.train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        for func in mappings:
            self.train = self.train.map(func)
            self.test = self.test.map(func)

        if shuffle:
            self.train = self.train.shuffle(self.batch_size * 64)
        buffer_size = 16
        self.train = self.train.batch(self.batch_size).prefetch(buffer_size)
        self.test = self.test.batch(self.batch_size).prefetch(buffer_size)

    def normalize_data(self):
        """
        normalize datasets according to mean, stddev
        :return: None
        """
        self.x_mean, self.x_std = du.get_mean_std(self.x_train)
        self.y_mean, self.y_std = du.get_mean_std(self.y_train)
        self.x_train = du.normalize(self.x_train, self.x_mean, self.x_std)
        self.y_train = du.normalize(self.y_train, self.y_mean, self.y_std)
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
