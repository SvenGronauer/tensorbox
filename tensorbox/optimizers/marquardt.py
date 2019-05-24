import tensorflow as tf
import abc


class UpdateMethod(abc.ABC):
    def __init__(self, name, loss_func, **kwargs):
        self.name = name
        self.loss_func = loss_func

    def get_config(self):
        return {
            'name': self.name,
        }

    @staticmethod
    def get_updates(batch, net):
        pass


class GradientDescent(UpdateMethod):
    """ Marquardt optimizer.

    """

    def __init__(self,
                 loss_func,
                 **kwargs):
        super(GradientDescent, self).__init__('GradientDescent', loss_func, **kwargs)

    def get_updates_and_loss(self, batch, net):
        data, label = batch
        with tf.GradientTape(persistent=False) as t:
            output = net(data)
            loss = self.loss_func(label, output)
        return t.gradient(loss, net.trainable_variables), loss

    def get_config(self):
        config = {
        }
        base_config = super(GradientDescent, self).get_config()
        base_config.update(config)
        return base_config


class MarquardtLevenberg(UpdateMethod):
    """ Marquardt Levenberg Method for optimizing.

    """

    def __init__(self,
                 loss_func,
                 constant=0.1,
                 **kwargs):
        super(MarquardtLevenberg, self).__init__(name='Marquardt',
                                                 loss_func=loss_func,
                                                 **kwargs)
        self.c = constant

    def get_updates_and_loss(self, batch, net):
        updates = []
        data, label = batch
        with tf.GradientTape(persistent=True) as t:
            output = net(data)
            loss = self.loss_func(label, output)
        dE_dw = t.gradient(loss, net.trainable_variables)
        dy_dw = t.jacobian(output, net.trainable_variables)
        del t  # free memory since gradient tape is persistent

        for i, (jac, gradient) in enumerate(zip(dy_dw, dE_dw)):

            assert len(jac.shape) == 4 or len(jac.shape) == 3, 'Unexpected shape of Jacobian.'
            if len(jac.shape) == 4:  # weights are of shape (n_batch, dim y, w_l-1, w_l)

                j_mean = tf.reduce_mean(jac, axis=0)  # take mean over batch
                shape_flat_weights = (-1, j_mean.shape[1] * j_mean.shape[2])
                j = tf.reshape(j_mean, shape_flat_weights)
                # @ is equivalent to tf.matmul()
                approximate_hessian = tf.transpose(j) @ j + self.c * tf.eye(j.shape[1])

                dw = tf.linalg.inv(approximate_hessian) @ tf.reshape(gradient, (-1, 1))
                dw = tf.reshape(dw, gradient.shape)  # flattened -> original shape of w
                assert dw.shape == gradient.shape, 'shapes do not match'
                updates.append(dw)

            elif len(jac.shape) == 3:  # biases are of shape (n_batch, dim y, w_l)

                # approx_hessian = tf.reduce_sum(tf.square(z) + lamb)
                # inv = 1. / approx_hessian

                j_mean = tf.reduce_mean(jac, axis=0)  # take mean over batch
                # TODO das hier kann kürzer geschrieben werden !!!
                approximate_hessian = tf.transpose(j_mean) @ j_mean + self.c * tf.eye(j_mean.shape[1])

                dw = tf.linalg.inv(approximate_hessian) @ tf.reshape(gradient, (-1, 1))
                dw = tf.reshape(dw, gradient.shape)  # flatten to original shape of w

                assert dw.shape == gradient.shape, 'shapes do not match'
                updates.append(dw)
            else:
                raise ValueError('Unexpected shape of Jacobian matrix.')
        return updates, loss

    def get_config(self):
        config = {
            'c': float(self.c)
        }
        base_config = super(MarquardtLevenberg, self).get_config()
        base_config.update(config)
        return base_config
        # return dict(list(base_config.items()) + list(config.items()))
