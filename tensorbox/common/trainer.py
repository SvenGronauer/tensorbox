import tensorflow as tf
from tensorflow.python import keras
import time
from abc import ABC, abstractmethod

import tensorbox.common.utils as utils


class Trainer(ABC):

    def __init__(self,
                 net,
                 opt,
                 log_path,
                 debug_level,
                 callbacks=None,
                 **kwargs):
        self.net = net
        self.opt = opt
        self.debug_level = debug_level
        if log_path is not None:
            self.log_path = log_path
            print('Logging into:', self.log_path)
        else:
            raise ValueError('log_path is None!')
        self.callbacks = callbacks

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              optimizer=opt,
                                              net=net)
        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                  self.log_path,
                                                  max_to_keep=5)

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)

    def predict(self, x):
        """  predict input according to network
        :param x: np.array or tf.tensor, input data
        :return: tf.Tensor(), holding the prediction of the input x
        """
        return self.net(x)

    def restore(self):
        """
        restore model from path
        :return: bool, true if restore is successful
        """
        restore_successful = False
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            restore_successful = True
        else:
            print("Initializing from scratch.")
        return restore_successful

    def save(self):
        utils.mkdir(self.log_path)
        self.checkpoint.step.assign_add(1)
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

    @abstractmethod
    def train(self, epochs):
        pass


class SupervisedTrainer(Trainer):
    def __init__(self,
                 net,
                 opt,
                 loss_func,
                 train_set,
                 test_set,
                 log_path,
                 debug_level=0,
                 **kwargs):
        super(SupervisedTrainer, self).__init__(net, opt, log_path, debug_level, **kwargs)
        assert isinstance(train_set, tf.data.Dataset), "Wrong format for dataset."

        self.train_set = train_set
        self.test_set = test_set
        self.loss_func = loss_func

        self.loss_metric = keras.metrics.Mean(name='test_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.restore()  # try to restore old checkpoints

    def train(self, epochs, metrics=None):
        for epoch in range(epochs):
            batch_losses = []
            batch_accs = []
            time_start = time.time()
            for i, batch in enumerate(self.train_set):
                batch_loss, batch_acc = self.train_step(batch)
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
            string = 'Epoch {} \t Loss: {:0.3f} \t Acc {:0.2f}% \t took {:0.2f}s'
            print(string.format(epoch,
                                utils.safe_mean(batch_losses),
                                utils.safe_mean(batch_accs),
                                time.time() - time_start))

    @tf.function
    def train_step(self, batch):
        image, label = batch
        with tf.GradientTape() as tape:
            predictions = self.net(image, training=True)
            loss = self.loss_func(label, predictions)
        gradients = tape.gradient(loss, self.net.trainable_variables)

        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result() * 100


class ReinforcementTrainer(Trainer, ABC):
    def __init__(self,
                 net,
                 opt,
                 env,
                 log_path,
                 debug_level,
                 **kwargs):
        super(ReinforcementTrainer, self).__init__(net, opt, log_path, debug_level, **kwargs)
        self.env = env
