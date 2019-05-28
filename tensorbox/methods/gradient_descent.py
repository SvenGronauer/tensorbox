from tensorbox.methods.basemethod import UpdateMethod


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