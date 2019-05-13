import tensorflow as tf
from tensorbox.datasets import get_dataset
from tensorflow.python import keras, layers

from tensorbox.networks.lenet import LeNet
from tensorbox.common.trainer import SupervisedTrainer

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


def create_interferer(filters=16, kernel_size=3, stride=1, pool_size=2):

    model = keras.Sequential()
    print(model)

    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))

    # mid-level
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))

    # up
    model.add(layers.Conv2DTranspose(filters, kernel_size, strides=pool_size, padding='same', use_bias=False))
    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(1, kernel_size, stride, padding='same', activation='tanh'))

    return model


def get_discriminator(train_ds, test_ds, log_path):

    net = LeNet(in_dim=(28, 28, 1), out_dim=10)
    opt = tf.keras.optimizers.Adam()
    loss_func = keras.losses.SparseCategoricalCrossentropy()

    return SupervisedTrainer(net, opt, loss_func, train_ds, test_ds, log_path)


def get_log_loss(y_true, y_pred, depth=10, eps=1.0e-9):
    """ get numerical stable log loss"""
    if len(y_true.shape) == 1:  # if labels are sparse: perform one-hot coding
        # depth = 1 + tf.cast(tf.reduce_max(y_true), tf.int32)
        y_true = tf.one_hot(y_true, depth=depth)

    y_pred_stable = tf.clip_by_value(y_pred, eps, 1.0)
    logloss = -1. * y_true * tf.math.log(y_pred_stable)

    weighted_loss_map = tf.reduce_sum(logloss, axis=-1)

    mean = tf.reduce_mean(weighted_loss_map)
    # w_mean = tf.reduce_mean(weighted_loss)

    return mean


def get_reward_signal(inputs, dicriminator):
    pass


def fast_gradient_sign_method(disc, image, label, eps=0.07):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = disc(image, training=True)
        loss = get_log_loss(label, predictions)
    grads = tape.gradient(loss, image)
    noise = tf.sign(grads) * eps
    return noise


def main(args):

    eta = 0.05
    dim = 3
    std_dev = 0.1
    log_path = '/var/tmp/delete_me'


    actions = tf.ones((dim, dim))
    exploration_noise = tf.random.normal(actions.shape, mean=0., stddev=std_dev)
    noisy_actions = actions + eta * exploration_noise
    ones = tf.ones((3, 3))

    fake = tf.random.normal([2, 28, 28, 1])

    interferer = create_interferer()

    train_ds, test_ds = get_dataset('mnist')

    discriminator = get_discriminator(train_ds, test_ds, log_path)

    for batch in test_ds:
        image, label = batch
        print(image.shape)
        pred = discriminator(image)
        loss = get_log_loss(label, pred)
        print('loss = {}'.format(loss.numpy()*100))

        noise = fast_gradient_sign_method(discriminator, image, label)
        pred = discriminator(image + noise)
        loss = get_log_loss(label, pred)
        print('loss after FGSM attack: {}'.format(loss.numpy()*100))

    # generated_image = interferer(fake, training=False)
    # print(generated_image.shape)


if __name__ == '__main__':
    main(args=None)

