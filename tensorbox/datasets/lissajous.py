#######################################
# Some words about this file
# created by
# Martin Gottwald, martin.gottwald@tum.de
# edited by
# Sven Gronauer, sven.gronauer@tum.de
#######################################


import tensorflow as tf
from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import DatasetWrapper
import tensorbox.datasets.data_utils as du

from matplotlib import pyplot as plt


class Lissajous:
    """  A regression dataset based on parameterized lissajous curves.
    The goal is to map a unit circle to an arbitrary Lissajous curve.
    The challenge is that the unit circle and the curves are different
    homeotopic types (depends on parameter choice)

    Created by Martin Gottwald, martin.gottwald@tum.de
    """
    def __init__(self, size, **kwargs):
        self.amplitude = (5.0, 5.0)   # amplitudes are simple scaling factors in x and y direction
        self.T = size
        self.dtype = kwargs.get("dtype", np.float64)

        # Scalar factor in front of the curve parameter t
        self.frequency = (2.0, 3.0)

        # Scalar bias which is added to the curve parameter t
        self.phase = (5.0 / 8.0 * np.pi, 0.0)

        # Std deviation for gaussian noise, currently 2.5% of amplitude for labels
        # and half the value for input samples
        self.stdX = 0.5 * 0.025 * max(self.amplitude)
        self.stdY = 0.025 * max(self.amplitude)

        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None

        self.create_data_set()

    def lissajous_curve(self, t):
        """ Evaluates the lissajous curves """
        return np.array([self.amplitude[0] * np.sin(self.frequency[0] * t + self.phase[0]),
                         self.amplitude[1] * np.sin(self.frequency[1] * t + self.phase[1])])

    def create_curve(self, T, allow_noise=False, offset=0.0):
        """ Call this function to create T samples and their labels """

        t = np.linspace(0.0+offset, 2.0 * np.pi+offset, T, dtype=self.dtype)

        X = np.array([np.cos(t), np.sin(t)]).T
        Y = self.lissajous_curve(t).T

        if allow_noise:
            X += np.random.normal(0.0, self.stdX, size=X.shape)
            Y += np.random.normal(0.0, self.stdY, size=Y.shape)

        return X, Y

    def create_data_set(self):
        ''' Implementation of the base class, here the actual data is created '''

        self.X_train, self.Y_train = self.create_curve(self.T, allow_noise=False, offset=0.0)
        self.X_val, self.Y_val = self.create_curve(self.T / 2, allow_noise=True)
        self.X_test, self.Y_test = self.create_curve(self.T, allow_noise=True, offset=.42)

    def plot_data_set(self, X, Y, run=None, window_title_suffix='-'):
        ''' Plots the given sorted data set'''

        from matplotlib import pyplot as plt
        # from matplotlib import patches

        fig = plt.gcf()

        fig.canvas.set_window_title("Lissajou data wrapper: {}".format(window_title_suffix))

        t = np.linspace(0, 1, X.shape[0])

        fig.add_subplot(121)
        plt.plot(X[:, 0], X[:, 1], '-k')
        # plt.scatter(X[:,0], X[:,1], c = t, cmap = 'hsv')
        plt.scatter(X[:, 0], X[:, 1], c=t, cmap='hsv')

        if run is not None:
            plt.title(run)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.grid()

        fig.add_subplot(122)
        plt.plot(Y[:, 0], Y[:, 1], '-k')
        plt.scatter(Y[:, 0], Y[:, 1], c=t, cmap='hsv')

        if run is not None:
            plt.title(run)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.grid()

    def plot_training_data_set(self, run=''):
        ''' Plots the current training data set. '''
        return self.plot_data_set(self.X_train,
                                  self.Y_train,
                                  run = run,
                                  window_title_suffix = "training data")

    def plot_test_data_set(self, run=''):
        ''' Plots the current training data set. '''
        return self.plot_data_set(self.X_test,
                                  self.Y_test,
                                  run=run,
                                  window_title_suffix="test data")

    def plot_regressor(self, run, regressor, batched=False, already_predicted=False):
        """ Plots the output of the regressor
        :param run:
        :param regressor:
        :param batched:
        :param already_predicted:
        :return:
        """

        if already_predicted:
            regressions = regressor
        else:
            if batched:
                regressions = regressor(self.X_test)
            else:
                regressions = np.apply_along_axis(regressor, 1, self.X_test)

        return self.plot_data_set(self.X_test,
                                  regressions,
                                  run=run,
                                  window_title_suffix="Regressor Output")


@DeprecationWarning
def change_type(data, label):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.float32)
    return data, label


def create_lissajous_dataset(batch_size=64, normalize=True, **kwargs):
    """ create dataset and wrap with tf.data.Dataset()
    :param batch_size:
    :param normalize: bool
    :param kwargs:
    :return:
    """
    lissa = Lissajous(size=1024)

    ds = DatasetWrapper(x_train=lissa.X_train,
                        y_train=lissa.Y_train,
                        x_test=lissa.X_test,
                        y_test=lissa.Y_test,
                        batch_size=batch_size,
                        wrapped_class=lissa,
                        mappings=(du.type_cast_sp, ),
                        name='Lissajous')
    ds.normalize_data() if normalize else None
    ds.normalize_labels() if normalize else None
    ds.build_tf_dataset()
    return ds


if __name__ == '__main__':
    print('testing..')
    lissa = create_lissajous_dataset()
    t = np.linspace(0, 1, lissa.wrapped_class.X_train.shape[0])
    # plt.figure()
    x1 = lissa.wrapped_class.X_train[:, 0]
    x2 = lissa.wrapped_class.X_train[:, 1]
    # plt.scatter(x1, x2, c=t, cmap='hsv')

    lissa.wrapped_class.plot_training_data_set()
    plt.show()



