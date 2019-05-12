import tensorflow as tf
from tensorflow.python import layers, keras
from tensorbox.networks import BaseNetwork


class LeNet(keras.Model, BaseNetwork):
    def __init__(self, in_dim, out_dim, filters=32, kernel_size=3, pool_size=2):
        super(LeNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fiters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.conv_1_1 = layers.Conv2D(filters, kernel_size, activation='relu')
        self.conv_1_2 = layers.Conv2D(filters, kernel_size, activation='relu')
        self.max_pool = layers.MaxPooling2D(pool_size, pool_size)
        self.conv_2_1 = layers.Conv2D(2*filters, kernel_size, activation='relu')
        self.conv_2_2 = layers.Conv2D(2*filters, kernel_size, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation='relu')
        # self.dropout = layers.Dropout(0.5)
        self.d2 = layers.Dense(out_dim, activation='softmax')

        self.init_weights_biases()

    def call(self, x, training=None, mask=None):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool(x)
        # second convolution block
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)

        # dense layers
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.dropout(x)
        return self.d2(x)

    def clone_net(self):
        """ creates a clone of the network model, but with different init values"""
        return LeNet(self.in_dim, self.out_dim, self.filters, self.kernel_size, self.pool_size)

    def get_config(self):
        raise NotImplementedError

    def init_weights_biases(self):
        """ perform forward-pass to create weights and biases"""
        fake_pass_shape = (1,) + self.in_dim
        self(tf.zeros(fake_pass_shape))
