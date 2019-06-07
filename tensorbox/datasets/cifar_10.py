import tensorflow as tf
from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import DatasetWrapper
import tensorbox.datasets.data_utils as du


def create_cifar_10_dataset(batch_size=64, normalize=True, mappings=(), **kwargs):
    """ create the Cifar-10 dataset and load into tf.data.Dataset()
    see http://www.cs.utoronto.ca/%7Ekriz/cifar.html
    :param batch_size:
    :param normalize:
    :param kwargs:
    :return:
    """
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

    ds = DatasetWrapper(x_train=train_x,
                        y_train=train_y,
                        x_test=test_x,
                        y_test=test_y,
                        batch_size=batch_size,
                        wrapped_class=None,
                        mappings=(du.type_cast_sp,),
                        name='Cifar-10')

    ds.normalize_data() if normalize else None
    ds.build_tf_dataset()
    return ds


if __name__ == '__main__':
    print('testing..')
    create_cifar_10_dataset()
