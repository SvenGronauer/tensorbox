import tensorflow as tf
from tensorbox.datasets.dataset_utils import convert_rgb_images_to_float
from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import Dataset


def change_type(data, label):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.float32)
    return data, label


def create_boston_dataset(train_val_split=0.8, batch_size=64, apply_preprocessing=True):
    test_split = 1. - train_val_split
    (train_x, train_y), (test_x, test_y) = datasets.boston_housing.load_data(test_split=test_split)
    mean = None
    std = None
    if apply_preprocessing:
        mean = np.mean(train_x, axis=0)
        std = np.std(train_x - mean, axis=0)

        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    buffer_size = batch_size * 16
    ds_train = ds_train.map(change_type).shuffle(buffer_size).batch(batch_size)
    ds_test = ds_test.map(change_type).batch(batch_size)
    x_shape = (13, )
    y_shape = (1,)
    return Dataset(train=ds_train,
                   test=ds_test,
                   mean=mean,
                   std=std,
                   x_shape=x_shape,
                   y_shape=y_shape)


if __name__ == '__main__':
    print('testing..')
