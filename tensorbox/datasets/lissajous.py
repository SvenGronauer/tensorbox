import tensorflow as tf
from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import DatasetWrapper
import tensorbox.datasets.data_utils as du
from tensorbox.datasets.datawrapper.lissajouswrapper import LissajousWrapper

from matplotlib import pyplot as plt


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
    lissa = LissajousWrapper(T=1024)

    ds = DatasetWrapper(x_train=lissa.X_train,
                        y_train=lissa.Y_train,
                        x_test=lissa.X_test,
                        y_test=lissa.Y_test,
                        batch_size=batch_size,
                        wrapped_class=lissa,
                        name='Lissajous')
    ds.normalize_data() if normalize else None
    ds.build_tf_dataset(mappings=(du.type_cast_sp, ))
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



