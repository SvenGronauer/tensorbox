import tensorflow as tf
from tensorflow.python import layers, keras

from tensorbox.networks.basenet import BaseNetwork
from tensorbox.common.classes import DatasetWrapper


class MLPNet(keras.Model, BaseNetwork):
    def __init__(self,
                 dataset=None,
                 in_dim=None,
                 out_dim=None,
                 units=(64, 64),
                 activation=tf.nn.relu,
                 **kwargs):
        assert isinstance(dataset, DatasetWrapper) or in_dim and out_dim, \
            'Please provide a valid data set or define in_dim and out_dim.'
        keras.Model.__init__(self)
        BaseNetwork.__init__(self, units=units, activation=activation, **kwargs)
        if dataset:
            self.in_dim = dataset.x_shape
            self.out_dim = dataset.y_shape
        else:
            self.in_dim = in_dim if isinstance(in_dim, tuple) else (in_dim, )
            self.out_dim = out_dim if isinstance(out_dim, tuple) else (out_dim, )

        hidden_units = list(units) + list(self.out_dim)
        self.dense_layers = []
        for n, n_units in enumerate(hidden_units):
            layer = layers.Dense(units=n_units,
                                 activation=activation if n < len(units) else None,
                                 name='dense_{}'.format(n))
            self.dense_layers.append(layer)

        self.init_weights_biases()

    def call(self, x, training=None, mask=None):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def clone_net_structure(self):
        """ creates a clone of the network model, but with different init values"""
        return MLPNet(in_dim=self.in_dim,
                      out_dim=self.out_dim,
                      units=self.units,
                      activation=self.activation)

    def get_config(self):
        return dict(units=self.units,
                    activation=self.activation,
                    in_dim=self.in_dim,
                    out_dim=self.out_dim)

    def init_weights_biases(self):
        """ perform forward-pass to init weights and biases"""
        fake_pass_shape = (1,) + self.in_dim
        self(tf.zeros(fake_pass_shape))
