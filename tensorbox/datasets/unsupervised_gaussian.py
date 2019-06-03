import tensorflow as tf
import numpy as np
from tensorbox.common.classes import DatasetWrapper
import tensorbox.datasets.data_utils as du
from tensorbox.datasets.datawrapper.lissajouswrapper import LissajousWrapper

from matplotlib import pyplot as plt


def change_type(data, label):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.float32)
    return data, label

def get_arrays(dim, size=1024, null_space=2, rotation=-45.0, axis_scale=(2., 1.)):
    """

    :param dim:
    :param size:
    :param null_space:
    :param rotation: float, rotation in degrees
    :param axis_scale:
    :return:
    """

    radian = rotation * np.pi / 180
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)],
                                [np.sin(radian), np.cos(radian)]])

    scale = np.array(axis_scale)

    train_x = np.zeros((size, null_space + dim))
    fake_y = np.zeros(size)

    array = np.matmul(np.random.normal(0.0, 1.0, (size, dim)) * scale, rotation_matrix)
    train_x[:, :dim] = array


    return train_x, fake_y

def create_unsupervised_gaussian_dataset(batch_size=64, dim=2, normalize=True, **kwargs):
    """ create dataset and wrap with tf.data.Dataset()
    :param batch_size:
    :param normalize: bool
    :param kwargs:
    :return:
    """
    train_x, fake_y = get_arrays(dim=dim)

    ds = DatasetWrapper(x_train=train_x,
                        y_train=train_x,
                        x_test=None,
                        y_test=None,
                        batch_size=batch_size,
                        wrapped_class=None,
                        name='Unsupervised Gaussian')
    ds.normalize_data() if normalize else None
    ds.build_tf_dataset(mappings=(du.type_cast_sp, ))
    return ds


if __name__ == '__main__':
    print('testing..')
    ds = create_unsupervised_gaussian_dataset(normalize=False)

    xs = ds.x_train[:, 0]
    ys = ds.x_train[:, 1]
    fig = plt.figure()

    u, s, v = np.linalg.svd(ds.x_train)

    plt.scatter(xs, ys)
    print('Right-sided vectors\n', v.T)
    plt.show()




