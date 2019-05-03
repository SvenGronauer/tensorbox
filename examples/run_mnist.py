import tensorflow as tf
from tensorflow.python import keras
import os

from toolbox.datasets import get_dataset
from toolbox.networks.lenet import LeNet, get_sequential_lenet
from toolbox.classes.trainer import SupervisedTrainer

import toolbox.common.utils as U


class MnistTrainer(SupervisedTrainer):

    def __init__(self, network, optimizer, loss, train_set, val_set, log_path, debug_level=0, **kwargs):
        super(MnistTrainer, self).__init__(network, optimizer, loss, train_set, log_path, debug_level, **kwargs)

        self.val_set = val_set

        self.loss_metric = tf.keras.metrics.Mean(name='test_loss')
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=network)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.log_path, max_to_keep=3)

        self.restore()

    def restore(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    @tf.function
    def train_step(self, batch):
        image, label = batch
        with tf.GradientTape() as tape:
            predictions = self.network(image)
            loss = self.loss(label, predictions)
        gradients = tape.gradient(loss, self.network.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result()*100

    def train(self, epochs, metrics=None):
        for epoch in range(epochs):
            batch_losses = []
            batch_accs = []
            for i, batch in enumerate(self.dataset):
                batch_loss, batch_acc = self.train_step(batch)
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
            print('Epoch {} \t Loss: {:0.3f} \t Acc {:0.2f}%'.format(epoch,
                                                                     U.safe_mean(batch_losses),
                                                                     U.safe_mean(batch_accs)))

    def save(self):

        U.mkdir(self.log_path)
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))


    def predict(self, x):
        return self.network(x)


def run_mnist(args):
    train_ds, val_ds = get_dataset('mnist')
    net = get_sequential_lenet()
    opt = tf.keras.optimizers.Adam()  # must be tf.keras.optimizers.Adam() not keras.optimizers.Adam()  !!!
    loss_func = keras.losses.SparseCategoricalCrossentropy()

    # network, optimizer, loss, train_set, val_set, log_path, debug_level = 0, ** kwargs
    trainer = MnistTrainer(network=net,
                           optimizer=opt,
                           loss=loss_func,
                           train_set=train_ds,
                           val_set=val_ds,
                           log_path=args.log_path)
    trainer.train(epochs=2)
    trainer.save()


if __name__ == '__main__':
    args = U.get_default_args()
    print(args)
    run_mnist(args)


