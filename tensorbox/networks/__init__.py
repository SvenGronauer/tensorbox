from abc import ABC, abstractmethod

""" convenient imports """
from tensorbox.networks.lenet import LeNet
from tensorbox.networks.shared_mlp import SharedMLPNet
from tensorbox.networks.mlp import MLPNet
from tensorbox.networks.unet import UNet


class BaseNetwork(ABC):
    """ Abstract base class that adds further functionality to tf.keras.Model """
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
        """ perform forward-pass to initialize weights and biases"""
        pass

