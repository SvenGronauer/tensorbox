import tensorflow as tf
from toolbox.datasets import get_dataset
from tensorflow.python import keras, layers
import numpy as np

from toolbox.networks.unet import UNet
from toolbox.classes.trainer import SupervisedTrainer

import matplotlib.pyplot as plt
import os

import toolbox.common.utils as U
import time
from toolbox.networks.lenet import LeNet


def create_interferer(filters=16, kernel_size=3, stride=1, pool_size=2):

    model = keras.Sequential()
    print(model)

    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))

    # mid-level
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))

    # up
    model.add(layers.Conv2DTranspose(filters, kernel_size, strides=pool_size, padding='same', use_bias=False))
    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(1, kernel_size, stride, padding='same', activation='tanh'))

    # assert model.output_shape == (None, 28, 28, 1)
    # assert model.output_shape == (None, 28, 28, 1)

    return model


def load_discriminator(ckpt_path, net, opt):

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        return True
    else:
        print("Initializing from scratch.")
        return False

    # new_model = tf.keras.models.load_model(ckpt_path)
    # return new_model


def main(args):
    fake = tf.random.normal([2, 28, 28, 1])

    interferer = create_interferer()

    disc = LeNet()
    disc_opt = tf.keras.optimizers.Adam()
    load_discriminator('/var/tmp/ga87zej/mnist/2019_05_02__19_06_11/',
                       net=disc,
                       opt=disc_opt)

    pred = disc(fake)
    print(pred.shape)

    generated_image = interferer(fake, training=False)

    print(generated_image.shape)
    # print(generated_image)


if __name__ == '__main__':
    main(args=None)

