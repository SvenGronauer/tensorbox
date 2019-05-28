import abc

""" convenient imports """
from tensorbox.methods.levenberg_marquardt import LevenbergMarquardt
from tensorbox.methods.gradient_descent import GradientDescent


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
