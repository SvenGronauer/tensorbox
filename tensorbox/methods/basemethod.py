from abc import ABC


class UpdateMethod(ABC):
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
