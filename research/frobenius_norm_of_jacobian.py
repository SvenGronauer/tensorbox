import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

""" tensorbox imports"""
from tensorbox.networks.mlp import MLPNet
from tensorbox.datasets import get_dataset

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


def fast_gradient_sign_method(batch, net, eps=0.05):
    data, label = batch
    with tf.GradientTape() as tape:
        tape.watch(data)
        predictions = net(data, training=False)
        loss = tf.reduce_sum(tf.square(label - predictions))
    grads = tape.gradient(loss, data)
    noise = tf.sign(grads) * eps
    return noise


def build_jacobian(net, x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = net(x)
    grads = t.gradient(y, x)
    return grads


def frobenius_norm_of_jacobi(net, x):
    jacobian = build_jacobian(net, x)
    return tf.reduce_sum(tf.square(jacobian))


def train_step_with_jacobian_norm(batch, net, opt, xi=0.5, **kwargs):
    data, label = batch
    with tf.GradientTape(persistent=True) as t:
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(data)
            y = net(data)
        jacobian = t2.gradient(y, data)
        # jacobian = t2.jacobian(y, data)
        mse = tf.reduce_mean(tf.square(y - label))
        norm = tf.reduce_sum(tf.square(jacobian))
        loss = mse + xi * norm
    grads = t.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss.numpy()
    # return grads


def train_step(batch, net, opt, **kwargs):
    data, label = batch
    with tf.GradientTape(persistent=True) as t:
        y = net(data)
        mse = tf.reduce_mean(tf.square(y - label))
    grads = t.gradient(mse, net.trainable_variables)
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


def create_adversarial_set(dataset, net, train_val_split=0.8):
    print('creating adversarial dataset...')
    adversarial_set = []
    labels = []
    for batch in dataset.test:
        data, label = batch
        noise = fast_gradient_sign_method(batch, net)
        new_data_sample = (data + noise).numpy()
        adversarial_set.append(new_data_sample)
        labels.append(label.numpy())
    adversarial_data = np.concatenate(tuple(adversarial_set), axis=0)
    adversarial_labels = np.concatenate(tuple(labels), axis=0)
    return tf.data.Dataset.from_tensor_slices((adversarial_data, adversarial_labels)).batch(64)


def train_network(dataset, net, opt, epochs, use_jacobian_norm=True, xi=0.2):

    train_func = train_step_with_jacobian_norm if use_jacobian_norm else train_step

    for epoch in range(epochs):
        losses = []
        for n_batch, batch in enumerate(dataset.train):

            loss = train_func(batch, net, opt, xi=xi)
            # print('loss with norm(J): ', loss)
            losses.append(loss)
    print('Epoch: {}  train loss: {:0.2f}  test loss: {:0.2f}'.format(epoch+1,
          np.mean(losses),
          evaluate(dataset.test, net)))


def train_and_test_adversarial(units, activation, xi):
    dataset = get_dataset('boston_housing')
    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)

    train_epochs = 150
    train_network(dataset, net, opt, epochs=train_epochs, use_jacobian_norm=True, xi=xi)
    adversarial_set = create_adversarial_set(dataset, net)
    eval_loss = evaluate(adversarial_set, net)
    print('Adversarial loss: {:0.4f}'.format(eval_loss))


def run_with_parameter_search():
    activations = [tf.nn.relu, tf.nn.tanh]
    units = [(64, 64), (100, 50, 25), (400, 300)]
    xis = [0.1, 0.5, 1.]

    for act in activations:
        for unit in units:
            for xi in xis:
                train_and_test_adversarial(unit, act, xi)
                break


if __name__ == '__main__':
    run_with_parameter_search()
    # main(args=None)

