import tensorflow as tf
from toolbox.datasets import get_dataset
import numpy as np

from toolbox.networks.unet import UNet
from toolbox.classes.trainer import SupervisedTrainer

import PIL
import imageio
import matplotlib.pyplot as plt
import os

import toolbox.common.utils as U
import time


def accuracy(y_true, y_pred, num_classes):
    return tf.reduce_mean(tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.float32) * y_true) * num_classes


def log_loss(y_true, y_pred):
    w = [1., 100., 100.]
    weights = tf.reshape(w, shape=(1, 1, 1, 3))
    logloss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weighted_loss = logloss * weights
    mean = tf.reduce_mean(logloss)
    w_mean = tf.reduce_mean(weighted_loss)
    return w_mean


class UNetTrainer(SupervisedTrainer):
    def __init__(self, network, optimizer, loss, train_set, val_set):
        super(UNetTrainer, self).__init__(network, optimizer, loss, train_set)
        self.val_set = val_set

        self.loss_metric = tf.keras.metrics.BinaryCrossentropy()
        self.acc = tf.metrics.BinaryAccuracy()

        self.callbacks = []
        self.callbacks_after_step = [self.callback_save_prediction]

    def callback_plot_progress(self, epoch):
        print('Epoch {}, Loss Metric: {:0.3f}\tAcc: {:0.3f}%'.format(epoch,
                                                                     self.loss_metric.numpy(),
                                                                     self.acc.numpy()*100))

    def callback_save_prediction(self, epoch):
        # TODO implement callback functions for trainers
        new = self.val_set
        for batch in new:
            image, gt = batch  # of shapes (batch_size, H, W, 3) and (batch_size, H, W, num_classes)
            pred = self.network(image)
            break

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(image[0, :, :])
        for i in range(pred.shape[3]):
            plt.subplot(2, 2, i + 2)
            plt.imshow(pred[0, :, :, i], cmap='gray', vmin=0., vmax=1.)
            if i == 0:
                plt.title('background')
            else:
                plt.title('class_'.format(i))
            plt.colorbar()

        # plt.show()
        U.mkdir('./tmp')
        fig.savefig('tmp/image_at_epoch_{:04d}.png'.format(epoch))

        """ ground truth """
        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(image[0, :, :])
        for i in range(gt.shape[3]):
            plt.subplot(2, 2, i + 2)
            plt.imshow(gt[0, :, :, i], cmap='gray', vmin=0., vmax=1.)
            if i == 0:
                plt.title('background')
            plt.colorbar()

        # plt.show()
        U.mkdir('./tmp')
        fig.savefig('tmp/gt_at_epoch_{:04d}.png'.format(epoch))

    def train(self, epochs):
        for epoch in range(epochs):
            ts = time.time()
            if epoch % 25 == 0:
                self.callback_save_prediction(epoch)
            for n, batch in enumerate(self.dataset):
                self.train_step(batch)
            self.callback_plot_progress(epoch)
            self.validate()
            print('took: {:0.2f}s'.format(time.time()-ts))

    # @tf.function
    def train_step(self, batch):
        images, labels = batch
        with tf.GradientTape() as tape:
            predictions = self.network(images)
            loss = self.loss_metric = log_loss(labels, predictions)
            self.acc = accuracy(y_true=labels, y_pred=predictions, num_classes=3)
            # loss = self.loss(labels, predictions, sample_weight=weights)
        gradients = tape.gradient(loss, self.network.trainable_variables)

        vars = self.network.trainable_variables
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        # self.loss_metric.update_state(labels, predictions)
        # self.acc.update_state(labels, predictions)


    def save(self):
        pass

    def predict(self, inputs):
        return self.network(inputs)

    def validate(self):
        mean_acc = []
        for n, batch in enumerate(self.val_set):
            image, gt = batch  # of shapes (batch_size, H, W, 3) and (batch_size, H, W, num_classes)
            pred = self.network(image)

            mean = tf.reduce_mean(tf.cast(tf.math.greater_equal(pred, 0.5), tf.float32) * gt) * self.network.num_classes
            mean_acc.append(mean.numpy())
        print('Validation mean: {:0.1f}%'.format(np.mean(mean_acc)*100))


if __name__ == '__main__':
    print(tf.__version__)
    # datasets = get_dataset('mnist', debug_level=1)
    train_ds, val_ds = get_dataset('tetris', debug_level=0)

    net = UNet(num_classes=3, num_filters=4, depth=3)
    vars = net.trainable_variables  # only available after forward pass !

    opt = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    # loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss_func = tf.losses.BinaryCrossentropy(from_logits=True)

    loss_func = tf.losses.LogLoss()

    trainer = UNetTrainer(network=net, optimizer=opt, loss=loss_func, train_set=train_ds, val_set=val_ds)
    trainer.train(epochs=201)

