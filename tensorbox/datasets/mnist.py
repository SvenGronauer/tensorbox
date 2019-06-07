import tensorflow as tf
import tensorbox.datasets.data_utils as du
from tensorbox.common.classes import DatasetWrapper


def create_mnist_dataset(train_val_split=0.8,
                         batch_size=32,
                         normalize=True,
                         **kwargs):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    ds = DatasetWrapper(x_train=train_images,
                        y_train=train_labels,
                        x_test=test_images,
                        y_test=test_labels,
                        batch_size=batch_size,
                        wrapped_class=None,
                        mappings=(du.convert_rgb_images_to_float, ),
                        name='MNIST',
                        **kwargs)

    ds.normalize_data() if normalize else None
    ds.build_tf_dataset()
    return ds


if __name__ == '__main__':
    print('testing..')
    create_mnist_dataset()
