import tensorflow as tf
from tensorbox.methods import UpdateMethod


class LevenbergMarquardt(UpdateMethod):
    """ Levenberg Marquardt method for second order optimization.
    """

    def __init__(self,
                 loss_func,
                 constant=0.1,
                 **kwargs):
        """ Constructor
        :param loss_func: tf.keras.losses, loss function to be applied
        :param constant: float, constant that is added for numerical stability (default: 0.1)
        :param kwargs:
        """
        super(LevenbergMarquardt, self).__init__(name='Marquardt',
                                                 loss_func=loss_func,
                                                 **kwargs)
        self.c = constant

    def get_updates_and_loss(self, batch, net):
        updates = []
        data, label = batch
        with tf.GradientTape(persistent=True) as t:
            output = net(data)
            loss = self.loss_func(label, output)
        de_dw = t.gradient(loss, net.trainable_variables)
        dy_dw = t.jacobian(output, net.trainable_variables)
        del t  # free memory since gradient tape is persistent

        for i, (jac, gradient) in enumerate(zip(dy_dw, de_dw)):

            assert len(jac.shape) == 4 or len(jac.shape) == 3, 'Unexpected shape of Jacobian.'
            j_mean = tf.reduce_mean(jac, axis=0)  # take mean over batch
            if len(jac.shape) == 4:  # weights are of shape (dim y, w_l-1, w_l)
                shape_flat_weights = (-1, j_mean.shape[1] * j_mean.shape[2])
                j = tf.reshape(j_mean, shape_flat_weights)
            else:  # biases are of shape (dim y, w_l)
                j = j_mean  # no reshape necessary
            # @ is equivalent to tf.matmul()
            approximate_hessian = tf.transpose(j) @ j + self.c * tf.eye(j.shape[1])
            # tf.linalg.solve() is approximately two times faster than tf.linalg.inv()
            dw = tf.linalg.solve(approximate_hessian, tf.reshape(gradient, (-1, 1)))
            dw = tf.reshape(dw, gradient.shape)  # flattened -> original shape of w
            assert dw.shape == gradient.shape, 'shapes do not match'
            updates.append(dw)

        return updates, loss

    def get_config(self):
        config = {
            'c': float(self.c)
        }
        base_config = super(LevenbergMarquardt, self).get_config()
        base_config.update(config)
        return base_config
