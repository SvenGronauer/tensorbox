from abc import ABC, abstractmethod


class BaseNetwork(ABC):

    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    @abstractmethod
    def clone_net(self):
        """ creates a clone of the network model, but with different init values"""
        pass

    @abstractmethod
    def get_config(self):
        return dict(units=self.units,
                    activation=self.activation)

    @abstractmethod
    def init_weights_biases(self):
        """ perform forward-pass to create weights and biases"""
        pass

