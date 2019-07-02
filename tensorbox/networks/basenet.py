from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    """ Abstract base class that adds further functionality to tf.keras.Model """
    def __init__(self, units, activation, **kwargs):
        self.units = units
        self.activation = activation
        self.kwargs = kwargs

    @abstractmethod
    def clone_net_structure(self):
        """ creates a clone of the network model, but with different init values"""
        pass

    @abstractmethod
    def get_config(self):
        return dict(units=self.units,
                    activation=self.activation)

    @abstractmethod
    def init_weights_biases(self):
        """ perform forward-pass to initialize weights and biases"""
        pass
