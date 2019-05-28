from tensorflow.python.keras import datasets
import numpy as np
from tensorbox.common.classes import DatasetWrapper
import tensorbox.datasets.data_utils as du


def create_boston_dataset(train_val_split=0.8, batch_size=32, normalize=True):
    test_split = 1. - train_val_split
    (train_x, train_y), (test_x, test_y) = datasets.boston_housing.load_data(test_split=test_split)

    train_y = np.reshape(train_y, (-1, 1))  # make labels two-dimensional: (None, 1)
    test_y = np.reshape(test_y, (-1, 1))

    ds = DatasetWrapper(x_train=train_x,
                        y_train=train_y,
                        x_test=test_x,
                        y_test=test_y,
                        batch_size=batch_size,
                        wrapped_class=None,
                        name='Boston Housing')
    if normalize:
        ds.normalize_data()
    ds.build_tf_dataset(mappings=(du.type_cast_sp, ))
    return ds


if __name__ == '__main__':
    print('testing..')
