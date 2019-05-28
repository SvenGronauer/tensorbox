import tensorflow as tf
from tensorflow.python import layers, keras

from tensorbox.networks.basenet import BaseNetwork


class SharedMLPNet(keras.Model, BaseNetwork):
    def __init__(self,
                 in_dim,
                 out_dims,
                 units=(64, 64),
                 activation=tf.nn.relu,
                 name='SharedMLP',
                 **kwargs):
        super(SharedMLPNet, self).__init__(name=name, **kwargs)
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.units = units
        self.activation = activation

        assert isinstance(out_dims, list) or isinstance(out_dims, tuple), \
            'out_dims must be a list or tuple object.'

        self.dense_layers = []
        for n, n_units in enumerate(units):
            layer = layers.Dense(units=n_units,
                                 activation=activation,
                                 name='dense_{}'.format(n))
            self.dense_layers.append(layer)
        self.heads = []
        for n, n_units in enumerate(out_dims):
            layer = layers.Dense(units=n_units,
                                 activation=None,  # heads always return logits
                                 name='head_{}'.format(n))
            self.heads.append(layer)

        self.init_weights_biases()

    def call(self, x, training=None, mask=None):
        for layer in self.dense_layers:
            x = layer(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs

    def clone_net(self):
        """ creates a clone of the network model, but with different init values"""
        return SharedMLPNet(self.in_dim,
                            self.out_dims,
                            units=self.units,
                            activation=self.activation)

    def get_config(self):
        raise NotImplementedError

    def init_weights_biases(self):
        """ perform forward-pass to create weights and biases"""
        fake_pass_shape = (1, ) + self.in_dim
        self(tf.zeros(fake_pass_shape))

