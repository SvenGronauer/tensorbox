import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python import keras

""" tensorbox imports"""
from tensorbox.networks import MLPNet
from tensorbox.datasets import get_dataset
from tensorbox.common import utils
from tensorbox.common.logger import CSVLogger
from tensorbox.common.classes import Configuration, BaseHook
from tensorbox.methods import GradientDescent
from tensorbox.common.trainer import SupervisedTrainer

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


class SVDHook(BaseHook):
    def __init__(self, net, log_dir):
        self.singular_values = []
        self.net = net
        self.index = 2
        self.log_dir = log_dir

    def final(self):
        self.save()

    def save(self):
        values = np.array(self.singular_values)
        file_path = os.path.join(self.log_dir, 'singular_values.csv')
        np.savetxt(file_path, values, delimiter=",")
        print('SVDHook saved to:', file_path)

    def hook(self):
        weights = self.net.get_weights()
        u, s, v = np.linalg.svd(weights[self.index])
        self.singular_values.append(s)
        # print(s.shape)


def main(args, **kwargs):
    units = (64, 64)
    activation = 'relu'
    dataset = get_dataset('lissajous')
    train_epochs = 200

    base_dir = '/var/tmp/ga87zej'
    log_dir = args.log_dir if args.log_dir else base_dir
    logger = CSVLogger(log_dir=log_dir, total_steps=train_epochs, stdout=True)
    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    lr = 1.0e-3
    opt = tf.keras.optimizers.Adam(lr=lr)

    metric = keras.metrics.Mean(name='mean_metric')
    loss_func = keras.losses.MeanSquaredError()
    method = GradientDescent(loss_func)

    config = Configuration(net=net,
                           opt=opt,
                           method=method,
                           dataset=dataset,
                           logger=logger,
                           log_dir=log_dir)

    svd_hook = SVDHook(net=net, log_dir=log_dir)
    trainer = SupervisedTrainer(loss_func=loss_func,
                                dataset=dataset,
                                metric=metric,
                                hooks=(svd_hook, ),
                                from_config=config)
    trainer.train(total_steps=15)

    # config.dump()


if __name__ == '__main__':
    args = utils.get_default_args()
    main(args)
    # main(args=None)

