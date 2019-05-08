from tensorflow.python import layers, keras


class SharedMLPNet(keras.Model):
    def __init__(self, out_dims, units=(64, 64), activation='relu'):
        super(SharedMLPNet, self).__init__()
        #  self.in_dim = in_dim  # todo define input dim!
        self.out_dims = out_dims

        assert isinstance(out_dims, list) or isinstance(out_dims, tuple), \
            'out_dims must be a list or tuple object.'

        # hidden_units = list(units) + [out_dim]
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

    def call(self, x, training=None, mask=None):
        for layer in self.dense_layers:
            x = layer(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs
