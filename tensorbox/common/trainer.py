import datetime
from abc import ABC, abstractmethod
import os
import tensorflow as tf

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
        # self.device = device
        if log_path is not None:
            self.log_path = log_path
            print('Logging into:', self.log_path)
        else:
            raise ValueError('log_path is None!')
        self.callbacks = callbacks

        # TODO add checkpoint manager
        self.manager = None
        self.checkpoint = None

    def predict(self, x):
        """
        predict input according to network
        :param x: np.array or tf.tensor, input data
        :return:
        """
        return self.net(x)

    def restore(self, path):
        """
        restore model from path
        :param path: str, path to checkpoint files
        :return: bool, true if restore is successful
        """
        restore_successful = False
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), 
                                         optimizer=self.opt,
                                         net=self.net)
        manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
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


class SupervisedTrainer(Trainer, ABC):
    def __init__(self, net, opt, loss_func, dataset, log_path, debug_level, **kwargs):
        super(SupervisedTrainer, self).__init__(net, opt, log_path, debug_level, **kwargs)

        if isinstance(dataset, tf.data.Dataset):
            self.dataset = dataset
        else:
            raise TypeError("Wrong format for dataset.")
        self.loss_func = loss_func


class ReinforcementTrainer(Trainer, ABC):
    def __init__(self, net, opt, env, log_path, debug_level, **kwargs):
        super(ReinforcementTrainer, self).__init__(net, opt, log_path, debug_level, **kwargs)
        self.env = env
