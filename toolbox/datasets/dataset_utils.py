import tensorflow as tf


def convert_rgb_images_to_float(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the images to [-1, 1]
    image = (image - 127.5) / 127.5

    return image, label

