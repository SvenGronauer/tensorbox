import tensorflow as tf
from tensorflow.python import keras, layers
from toolbox.datasets import get_dataset
import numpy as np

from toolbox.networks.unet import UNet
from toolbox.classes.trainer import SupervisedTrainer

import matplotlib.pyplot as plt
import os

import toolbox.common.utils as U
import time


def accuracy(y_true, y_pred, num_classes):
    return tf.reduce_mean(tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.float32) * y_true) * num_classes


def log_loss(y_true, y_pred):
    w = [1., 25., 50.]
    weights = tf.reshape(w, shape=(1, 1, 1, 3))
    # logloss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    logloss = - y_true * tf.math.log(y_pred) * weights

    weighted_loss_map = tf.reduce_sum(logloss, axis=3)

    # weighted_loss = logloss * weights
    mean = tf.reduce_mean(weighted_loss_map)
    # w_mean = tf.reduce_mean(weighted_loss)
    return mean


class UNetTrainer(SupervisedTrainer):
    def __init__(self, network, optimizer, loss, train_set, val_set, log_path, debug_level=0, **kwargs):
        super(UNetTrainer, self).__init__(network, optimizer, loss, train_set, log_path, debug_level, **kwargs)
        self.val_set = val_set

        # self.loss_metric = keras.metrics.BinaryCrossentropy()
        self.loss_metric = keras.metrics.CategoricalCrossentropy()
        # self.acc = tf.metrics.BinaryAccuracy()
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
        U.mkdir(self.log_path)
        # figure_name_path = self.log_path + '/image_at_epoch_{:04d}.png'.format(epoch)
        figure_name_path = os.path.join(self.log_path, 'image_at_epoch_{:04d}.png'.format(epoch))
        fig.savefig(figure_name_path)

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
        U.mkdir(self.log_path)
        figure_name_path = os.path.join(self.log_path, 'gt_at_epoch_{:04d}.png'.format(epoch))
        fig.savefig(figure_name_path)

    def train(self, epochs):
        for epoch in range(epochs):
            ts = time.time()
            if epoch % 20 == 0:
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


def main(args):
    print(tf.__version__)
    # datasets = get_dataset('mnist', debug_level=1)
    train_ds, val_ds = get_dataset('tetris', debug_level=0)

    net = UNet(num_classes=3, num_filters=8, depth=3)
    vars = net.trainable_variables  # only available after forward pass !

    opt = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    # loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss_func = tf.losses.BinaryCrossentropy(from_logits=True)

    loss_func = tf.losses.LogLoss()

    trainer = UNetTrainer(network=net,
                          optimizer=opt,
                          loss=loss_func,
                          train_set=train_ds,
                          val_set=val_ds,
                          log_path=args.log_path)
    trainer.train(epochs=201)


if __name__ == '__main__':
    main(args=None)

