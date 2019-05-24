import tensorflow as tf
from tensorflow.python import layers, keras
from tensorbox.networks import BaseNetwork


class MLPNet(keras.Model, BaseNetwork):
    def __init__(self, in_dim, out_dim, units=(64, 64), activation=tf.nn.relu):
        keras.Model.__init__(self)
        BaseNetwork.__init__(self, units=units, activation=activation)
        # super(MLPNet, self).__init__(units=units, activation=activation)
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_units = list(units) + list(out_dim)
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

    def clone_net(self):
        """ creates a clone of the network model, but with different init values"""
        return MLPNet(self.in_dim,
                      self.out_dims,
                      units=self.units,
                      activation=self.activation)

    def get_config(self):
        return dict(units=self.units,
                    activation=self.activation,
                    in_dim=self.in_dim,
                    out_dim=self.out_dim)

    def init_weights_biases(self):
        """ perform forward-pass to create weights and biases"""
        fake_pass_shape = (1,) + self.in_dim
        self(tf.zeros(fake_pass_shape))
