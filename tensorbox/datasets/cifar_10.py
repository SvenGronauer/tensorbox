import tensorflow as tf
from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import Dataset, DatasetWrapper
import tensorbox.datasets.data_utils as du


def create_cifar_10_dataset(batch_size=64, apply_preprocessing=True, **kwargs):
    """ create the Cifar-10 dataset and load into tf.data.Dataset()
    see http://www.cs.utoronto.ca/%7Ekriz/cifar.html
    :param batch_size:
    :param apply_preprocessing:
    :param kwargs:
    :return:
    """
    # test_split = 1. - train_val_split
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds_train = ds_train.map(du.convert_rgb_images_to_float)
    ds_train = ds_train.batch(batch_size).prefetch(16)

    ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds_test = ds_test.map(du.convert_rgb_images_to_float)
    ds_test = ds_test.batch(batch_size)

    x_shape = train_x.shape[1:]
    y_shape = train_y.shape[1:]

    return DatasetWrapper(train=ds_train,
                          test=ds_test,
                          x_shape=x_shape,
                          y_shape=y_shape)


if __name__ == '__main__':
    print('testing..')
    create_cifar_10_dataset()
