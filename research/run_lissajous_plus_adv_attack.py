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


def train_step_with_jacobian_norm(batch, net, opt, xi=0.001, **kwargs):
    data, label = batch
    with tf.GradientTape(persistent=True) as t:
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(data)
            y = net(data)
        jacobian = t2.gradient(y, data)
        # jacobian = t2.jacobian(y, data)
        mse = tf.reduce_mean(tf.square(y - label))
        norm = tf.reduce_mean(tf.square(jacobian))
        loss = mse + xi * norm
    grads = t.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss.numpy()


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


def train_network(dataset, net, opt, epochs, use_frobenius_norm):

    train_func = train_step_with_jacobian_norm if use_frobenius_norm else train_step_normal


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


def fast_gradient_sign_method(batch, net, eps=0.05):
    data, label = batch
    with tf.GradientTape() as tape:
        tape.watch(data)
        predictions = net(data, training=False)
        loss = tf.reduce_sum(tf.square(label - predictions))
    grads = tape.gradient(loss, data)
    noise = tf.sign(grads) * eps
    return noise


def normed_gradient(batch, net, eps=0.05):
    data, label = batch
    with tf.GradientTape() as tape:
        tape.watch(data)
        predictions = net(data, training=False)
        loss = tf.reduce_sum(tf.square(label - predictions))
    grads = tape.gradient(loss, data)
    # norm_factor = tf.divide(1., tf.norm(grads, axis=1))
    # noise = tf.sign(grads) * eps
    scale = 0.5
    normed_grads = scale * grads # * norm_factor
    return normed_grads


def create_adversarial_from_test_set(dataset, net):
    print('creating adversarial dataset...')
    adversarial_set = []
    labels = []
    fast_gradients = []
    for batch in dataset.test:
        data, label = batch
        # fast_gradient_sign = fast_gradient_sign_method(batch, net)
        fast_gradient_sign = normed_gradient(batch, net)
        fast_gradients.append(fast_gradient_sign.numpy())
        new_data_sample = (data + fast_gradient_sign).numpy()
        adversarial_set.append(new_data_sample)
        labels.append(label.numpy())
    adversarial_data = np.concatenate(tuple(adversarial_set), axis=0)
    adversarial_labels = np.concatenate(tuple(labels), axis=0)

    adversarial_set = tf.data.Dataset.from_tensor_slices((adversarial_data,
                                                          adversarial_labels)).batch(64)
    fast_gradients = np.array(fast_gradients)
    fast_gradients = fast_gradients.reshape((-1, ) + fast_gradients.shape[2:])
    return adversarial_set, fast_gradients


def main(args, **kwargs):
    activation = tf.nn.tanh
    units = (100, 50, 25)
    dataset = get_dataset('lissajous')

    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    # opt = tf.keras.optimizers.SGD(lr=1.0e-3)

    train_epochs = 50
    train_network(dataset, net, opt, epochs=train_epochs, use_frobenius_norm=True)

    adversarial_data_set, fast_gradients = create_adversarial_from_test_set(dataset, net)
    test_set_loss = evaluate(dataset.test, net)
    print('Test Set Loss: {:0.4f}'.format(test_set_loss))
    eval_loss = evaluate(adversarial_data_set, net)
    print('Adversarial loss: {:0.4f}'.format(eval_loss))
    plt.figure(), dataset.plot()

    plt.figure(), plot_vectors(dataset, fast_gradients)

    plt.figure(), dataset.plot_predictions(net(dataset.x_test).numpy())
    plt.show()


def plot_vectors(dataset, fast_gradients):

    N = 64
    scale = 5.

    xs = dataset.x_test[::N, 0]
    ys = dataset.x_test[::N, 1]

    x_vec = fast_gradients[::N, 0] * scale
    y_vec = fast_gradients[::N, 1] * scale

    plt.quiver(xs, ys, x_vec, y_vec, angles="xy", scale_units="xy", scale=1)
    # plt.show()


if __name__ == '__main__':
    args = utils.get_default_args()
    main(args)
    # plot_vectors()
    # main(args=None)

