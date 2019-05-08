from tensorflow.python import layers, keras


class MLPNet(keras.Model):
    def __init__(self, out_dim, units=(64, 64), activation='relu'):
        super(MLPNet, self).__init__()
        #  self.in_dim = in_dim  # todo define input dim!
        self.out_dim = out_dim

        hidden_units = list(units) + [out_dim]
        self.dense_layers = []
        for n, n_units in enumerate(hidden_units):
            layer = layers.Dense(units=n_units,
                                 activation=activation if n < len(units) else None,
                                 name='dense_{}'.format(n))
            self.dense_layers.append(layer)

    def call(self, x, training=None, mask=None):
        for layer in self.dense_layers:
            x = layer(x)
        return x
