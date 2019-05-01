import datetime
from abc import ABC, abstractmethod
import os
import tensorflow as tf


class Trainer(ABC):

    def __init__(self, optimizer, callbacks=None):
        self.optimizer = optimizer
        # self.device = device
        self.log_path = '/var/tmp/' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
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
    def __init__(self, network, optimizer, loss, dataset, device=None):
        super(SupervisedTrainer, self).__init__(optimizer, device)
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
