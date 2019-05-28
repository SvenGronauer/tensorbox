import tensorflow as tf
from tensorbox.methods.basemethod import UpdateMethod


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
        dy_dw = t.gradient(output, net.trainable_variables)  # equals the sum over batch
        # dy_dw = t.jacobian(output, net.trainable_variables)  # BUG - RAM increases until crash
        del t  # free memory since gradient tape is persistent

        for i, (jac, gradient) in enumerate(zip(dy_dw, de_dw)):

            assert len(jac.shape) == 1 or len(jac.shape) == 2, 'Unexpected shape of Jacobian.'
            if len(jac.shape) == 2:  # weights are of shape (w_l-1, w_l)
                flat_weights_shape = (-1, jac.shape[0] * jac.shape[1])
                j = tf.reshape(jac, flat_weights_shape)
            else:  # biases are of shape (w_l)
                j = tf.reshape(jac, (-1, jac.shape[0]))  # flatten

            # @ is equivalent to tf.matmul()
            approximate_hessian = tf.transpose(j) @ j + self.c * tf.eye(j.shape[1])
            dw = tf.linalg.solve(approximate_hessian, tf.reshape(gradient, (-1, 1)))
            dw = tf.reshape(dw, gradient.shape)  # flatten -> original shape of w
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
