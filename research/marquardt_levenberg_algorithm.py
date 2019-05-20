import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np

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
        err = y - label
        mse = tf.reduce_mean(tf.square(err))

    grads = t.gradient(mse, net.trainable_variables)

    opt.apply_gradients(zip(grads, net.trainable_variables))
    # return 0
    return mse.numpy()
    # return grads


# @tf.function
def train_step_with_marquardt_levenberg(batch, net, opt, **kwargs):
    data, label = batch
    with tf.GradientTape(persistent=True) as t:

        y = net(data)
        err = y - label
        mse = tf.reduce_mean(tf.square(err))

    grads = []
    lamb = 0.1
    dE_dw = t.gradient(mse, net.trainable_variables)
    J = t.gradient(y, net.trainable_variables)
    for i, z in enumerate(J):

        if len(z.shape) == 2:  # for weights
            dim = z.shape[1]
            approx_hessian = tf.matmul(tf.transpose(z), z) + lamb * tf.eye(dim)
            inv = tf.linalg.inv(approx_hessian)
            gradient = dE_dw[i]
            # dw = tf.matmul(inv, tf.transpose(z))
            dw = tf.matmul(inv, tf.transpose(gradient))
            dw = tf.transpose(dw)

            assert dw.shape == gradient.shape, 'shapes do not match'
            grads.append(dw)
        elif len(z.shape) == 1:  # for biases
            gradient = dE_dw[i]
            approx_hessian = tf.reduce_sum(tf.square(z) + lamb)
            inv = 1. / approx_hessian
            dw = gradient * inv
            assert dw.shape == gradient.shape, 'shapes do not match'
            dw = gradient  # TODO change me
            grads.append(dw)

    del t  # free memory
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return mse.numpy()
    # return grads


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
    return np.mean(losses)


def train_network(dataset, net, opt, epochs, use_marquardt_levenberg):

    # train_func = train_step_with_jacobian_norm if use_jacobian_norm else train_step
    train_func = train_step_with_marquardt_levenberg if use_marquardt_levenberg else train_step_normal

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
    activation = tf.nn.tanh
    units = (64, 64)
    dataset = get_dataset('boston_housing')

    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    # opt = tf.keras.optimizers.SGD(lr=1.0e-3)

    train_epochs = 150
    train_network(dataset, net, opt, epochs=train_epochs, use_marquardt_levenberg=False)


if __name__ == '__main__':
    args = utils.get_default_args()
    main(args)
    # main(args=None)

