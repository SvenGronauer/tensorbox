import tensorflow as tf
from toolbox.datasets import get_dataset

from toolbox.networks.unet import UNet


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


if __name__ == '__main__':
    print(tf.__version__)
    # datasets = get_dataset('mnist', debug_level=1)
    train_ds, val_ds = get_dataset('mnist', debug_level=1)

    net = UNet(num_filters=8, num_classes=3)
    opt = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    trainer = Trainer(network=net, optimizer=opt, loss_func=loss_func)
    trainer.train(dataset=train_ds)

