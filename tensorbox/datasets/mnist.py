import tensorflow as tf
from tensorbox.datasets.dataset_utils import convert_rgb_images_to_float


def create_mnist_dataset(train_val_split=0.8, batch_size=32, apply_preprocessing=True):
    (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(-1, 28, 28, 1)
    val_images = val_images.reshape(-1, 28, 28, 1)

    # convert labels into one-hot vectors  -> no need for that: use sparse categorial loss
    # train_labels = tf.one_hot(train_labels, depth=10)
    # val_labels = tf.one_hot(val_labels, depth=10)

    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    if apply_preprocessing:
        buffer_size = batch_size * 16
        ds_train = ds_train.map(convert_rgb_images_to_float).shuffle(buffer_size).batch(batch_size)
        ds_val = ds_val.map(convert_rgb_images_to_float).batch(batch_size)
    return ds_train, ds_val


if __name__ == '__main__':
    print('testing..')
