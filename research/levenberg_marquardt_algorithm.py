import tensorflow as tf
import numpy as np
import time
from tensorflow.python import keras

""" tensorbox imports"""
from tensorbox.networks import MLPNet
from tensorbox.datasets import get_dataset
from tensorbox.common import utils
from tensorbox.common.logger import CSVLogger
from tensorbox.common.classes import Configuration
from tensorbox.methods import LevenbergMarquardt, GradientDescent

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


def build_jacobian(net, x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = net(x)
    grads = t.gradient(y, x)
    return grads


def evaluate(test_set, net, debug=False):
    losses = []
    for n_batch, batch in enumerate(test_set):
        data, label = batch
        y = net(data, training=False)
        mse = tf.reduce_mean(tf.square(y - label))
        losses.append(mse.numpy())
        if debug:
            print('Labels')
            print(label)
    return float(np.mean(losses))


def train_network(dataset, net, opt, method, epochs, logger=None):

    loss_metric = keras.metrics.Mean(name='mean')
    losses = []
    for epoch in range(1, epochs+1):
        losses = []
        ts = time.time()
        for n_batch, batch in enumerate(dataset.train):
            updates, loss = method.get_updates_and_loss(batch, net)
            loss_value = loss_metric(loss).numpy()
            opt.apply_gradients(zip(updates, net.trainable_variables))
            losses.append(loss_value)

        if logger:
            write_dic = dict(loss_train=utils.safe_mean(losses),
                             loss_test=evaluate(dataset.test, net),
                             time=time.time()-ts)
            logger.write(write_dic, epoch)


def main(args, units, activation, use_marquardt=True, **kwargs):
    dataset = get_dataset('boston_housing')

    base_dir = '/var/tmp/ga87zej'
    log_dir = args.log_dir if args.log_dir else base_dir
    logger = CSVLogger(log_dir, stdout=False)
    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    # opt = tf.keras.methods.Adam(lr=1.0e-3)
    lr = 1.0e-3
    opt = tf.keras.optimizers.SGD(lr=lr) if use_marquardt else tf.keras.optimizers.Adam(lr=lr)

    loss_func = keras.losses.MeanSquaredError()

    method = LevenbergMarquardt(loss_func) if use_marquardt else GradientDescent(loss_func)

    config = Configuration(net=net,
                           opt=opt,
                           method=method,
                           log_dir=log_dir)
    config.dump()
    train_epochs = 150
    train_network(dataset,
                  net,
                  opt,
                  method=method,
                  epochs=train_epochs,
                  logger=logger)


def param_search():
    list_units = [(8, 8), (16, 16), (32, 32)]
    list_activations = ['relu', 'tanh']
    modes = [True, False]
    runs_per_setting = 5

    for units in list_units:
        for activation in list_activations:
            for use_marquardt in modes:
                for i in range(runs_per_setting):
                    args = utils.get_default_args()
                    print('============================================================')
                    print('Units: {}, Activation: {}, Marquardt? {}'.format(units,
                                                                            activation,
                                                                            use_marquardt))
                    main(args, units=units, activation=activation, use_marquardt=use_marquardt)
            return 0


if __name__ == '__main__':
    # args = utils.get_default_args()
    # main(args)
    param_search()
    # main(args=None)

