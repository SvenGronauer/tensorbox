import tensorflow as tf
import sys

from toolbox.datasets import get_dataset
from toolbox.networks.lenet import LeNet


class Trainer:
    def __init__(self, network, optimizer, loss_func):
        self.network = network
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_metric = tf.keras.metrics.Mean(name='test_loss')
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.network(image)
            loss = self.loss_func(label, predictions)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        # print(gradients.shape)
        vars = self.network.trainable_variables
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result()*100

    def train(self, dataset, metrics=None):
        for i, batch in enumerate(dataset):
            image, label = batch
            print(i, image.shape)
            self.train_step(image, label)


def run_mnist(args):
    train_ds, val_ds = get_dataset('mnist', debug_level=args.debug)
    net = LeNet()
    opt = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    trainer = Trainer(network=net, optimizer=opt, loss_func=loss_func)
    trainer.train(dataset=train_ds)


if __name__ == '__main__':
    print(tf.__version__)
    run_mnist(sys.argv)
    # datasets = get_dataset('mnist', debug_level=1)


