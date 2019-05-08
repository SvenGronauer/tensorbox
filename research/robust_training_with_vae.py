import tensorflow as tf
from tensorbox.datasets import get_dataset
from tensorflow.python import keras, layers

from tensorbox.networks.lenet import LeNet


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

    # assert model.output_shape == (None, 28, 28, 1)
    # assert model.output_shape == (None, 28, 28, 1)

    return model


def load_discriminator(ckpt_path, net, opt):

    restore_sucessfull = False

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        restore_sucessfull = True
    else:
        print("Initializing from scratch.")
    return restore_sucessfull


def get_log_loss(y_true, y_pred, eps=1.0e-10):

    if len(y_true.shape) == 1:  # perform one-hot coding of labels
        depth = 1 + tf.cast(tf.reduce_max(y_true), tf.int32)
        y_true = tf.one_hot(y_true, depth=depth)

    y_pred_stable = tf.clip_by_value(y_pred, eps, 1.0)
    logloss = -1. * y_true * tf.math.log(y_pred_stable)

    weighted_loss_map = tf.reduce_sum(logloss, axis=-1)

    mean = tf.reduce_mean(weighted_loss_map)
    # w_mean = tf.reduce_mean(weighted_loss)

    return mean


def get_reward_signal(inputs, dicriminator):
    pass


def main(args):

    eta = 0.05
    dim = 3
    std_dev = 0.1

    train_ds, test_ds = get_dataset('mnist')

    actions = tf.ones((dim, dim))
    exploration_noise = tf.random.normal(actions.shape, mean=0., stddev=std_dev)
    noisy_actions = actions + eta * exploration_noise
    ones = tf.ones((3, 3))

    fake = tf.random.normal([2, 28, 28, 1])

    interferer = create_interferer()

    value_net = LeNet(out_dim=1)

    disc = LeNet(out_dim=10)
    disc_opt = tf.keras.optimizers.Adam()
    load_discriminator('/Users/sven/git/tensorbox/tmp/mnist/2019_05_06__13_43_47',
                       net=disc,
                       opt=disc_opt)

    for batch in test_ds:
        image, label = batch
        print(image.shape)
        pred = disc(image)
        loss = get_log_loss(label, pred)
        break
    print('loss =', loss.numpy())
    pred = disc(fake)
    print(pred.shape)

    generated_image = interferer(fake, training=False)

    print(generated_image.shape)
    # print(generated_image)


if __name__ == '__main__':
    main(args=None)

