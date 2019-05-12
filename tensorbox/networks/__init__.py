from abc import ABC, abstractmethod


class BaseNetwork(ABC):

    @abstractmethod
    def clone_net(self):
        """ creates a clone of the network model, but with different init values"""
        pass

    @abstractmethod
    def init_weights_biases(self):
        """ perform forward-pass to create weights and biases"""
        pass
