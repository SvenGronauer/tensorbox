import datetime
from abc import ABC, abstractmethod
import os
import tensorflow as tf


class Trainer(ABC):

    def __init__(self, optimizer, log_path, debug_level, callbacks=None, **kwargs):
        self.optimizer = optimizer
        self.debug_level = debug_level
        # self.device = device
        if log_path is not None:
            self.log_path = log_path
        else:
            raise ValueError('log_path is None!')
        self.callbacks = callbacks

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def train(self, epochs):
        pass


class SupervisedTrainer(Trainer, ABC):
    def __init__(self, network, optimizer, loss, dataset, log_path, debug_level, **kwargs):
        super(SupervisedTrainer, self).__init__(optimizer, log_path, debug_level)
        self.network = network

        if isinstance(dataset, tf.data.Dataset):
            self.dataset = dataset
        else:
            raise TypeError("Wrong format for dataset.")
        self.loss = loss


class ReinforcementTrainer(Trainer, ABC):
    def __init__(self, network, env, optimizer, device):
        super(ReinforcementTrainer, self).__init__(optimizer, device)
        self.network = network
        self.env = env
