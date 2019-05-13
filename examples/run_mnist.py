import tensorflow as tf
from tensorflow.python import keras
import time

from tensorbox.datasets import get_dataset
from tensorbox.networks.lenet import LeNet
from tensorbox.common.trainer import SupervisedTrainer
import tensorbox.common.utils as utils

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


class MnistTrainer(SupervisedTrainer):
    def __init__(self,
                 net,
                 opt,
                 loss_func,
                 train_set,
                 val_set,
                 log_path,
                 debug_level=0,
                 **kwargs):
        super(MnistTrainer, self).__init__(net, opt, loss_func, train_set,
                                           log_path, debug_level, **kwargs)

        self.val_set = val_set

        self.loss_metric = keras.metrics.Mean(name='test_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
        # self.manager = tf.train.CheckpointManager(self.ckpt, self.log_path,
        #                                           max_to_keep=3)

        self.restore()  # try to restore old checkpoints

    @tf.function
    def train_step(self, batch):
        image, label = batch
        with tf.GradientTape() as tape:
            predictions = self.net(image, training=True)
            loss = self.loss_func(label, predictions)
        gradients = tape.gradient(loss, self.net.trainable_variables)

        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result() * 100

    def train(self, epochs, metrics=None):
        for epoch in range(epochs):
            batch_losses = []
            batch_accs = []
            time_start = time.time()
            for i, batch in enumerate(self.dataset):
                batch_loss, batch_acc = self.train_step(batch)
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
            string = 'Epoch {} \t Loss: {:0.3f} \t Acc {:0.2f}% \t took {:0.2f}s'
            print(string.format(epoch,
                                utils.safe_mean(batch_losses),
                                utils.safe_mean(batch_accs),
                                time.time() - time_start))


def run_mnist(args):
    train_ds, val_ds = get_dataset('mnist')
    net = LeNet(in_dim=(28, 28, 1), out_dim=10)
    opt = tf.keras.optimizers.Adam()  # must be tf.keras.optimizers.Adam() not keras.optimizers.Adam()  !!!
    loss_func = keras.losses.SparseCategoricalCrossentropy()

    trainer = MnistTrainer(net,
                           opt,
                           loss_func,
                           train_set=train_ds,
                           val_set=val_ds,
                           log_path=args.log_path)
    try:
        trainer.train(epochs=50)
    except KeyboardInterrupt:
        print('got KeyboardInterrupt')
    finally:
        trainer.save()


if __name__ == '__main__':
    args = utils.get_default_args()
    print(args)
    run_mnist(args)

