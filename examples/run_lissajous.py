import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import matplotlib.pyplot as plt

""" tensorbox imports"""
from tensorbox.networks.mlp import MLPNet
from tensorbox.datasets import get_dataset
from tensorbox.common import utils

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


def build_jacobian(net, x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = net(x)
    grads = t.gradient(y, x)
    return grads


# @tf.function
def train_step_normal(batch, net, opt, **kwargs):
    data, label = batch
    with tf.GradientTape(persistent=False) as t:
        y = net(data)
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(y - label), axis=1))
    grads = t.gradient(mse, net.trainable_variables)

    opt.apply_gradients(zip(grads, net.trainable_variables))
    # return 0
    return mse.numpy()
    # return grads


def evaluate(test_set, net, debug=False):
    losses = []
    for n_batch, batch in enumerate(test_set):
        data, label = batch
        y = net(data, training=False)
        square = tf.reduce_sum(tf.square(y - label), axis=1)
        mse = tf.reduce_mean(square)
        losses.append(mse.numpy())

    return utils.safe_mean(losses)


def train_network(dataset, net, opt, epochs, use_marquardt_levenberg):

    # train_func = train_step_with_jacobian_norm if use_jacobian_norm else train_step
    train_func = train_step_normal

    losses = []
    for epoch in range(epochs):
        losses = []
        for n_batch, batch in enumerate(dataset.train):

            loss = train_func(batch, net, opt)
            # print('loss with norm(J): ', loss)
            losses.append(loss)
        print('Epoch: {}  train loss: {:0.3f}  test loss: {:0.3f}'.format(epoch+1,
          np.mean(losses),
          evaluate(dataset.test, net)))


def main(args, **kwargs):
    activation = tf.nn.relu
    units = (64, 64)
    dataset = get_dataset('lissajous')

    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    # opt = tf.keras.methods.SGD(lr=1.0e-3)

    train_epochs = 50
    train_network(dataset, net, opt, epochs=train_epochs, use_marquardt_levenberg=False)

    plt.figure(), dataset.plot()

    # y_pred = net(dataset.x_train).numpy()
    plt.figure(), dataset.plot_predictions(net(dataset.x_test).numpy())
    plt.show()


if __name__ == '__main__':
    args = utils.get_default_args()
    main(args)
    # main(args=None)

