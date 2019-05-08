from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class ProbabilityDistribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def entropy(self):
        pass

    @abstractmethod
    def neg_log_prob(self):
        pass

    @abstractmethod
    def sample(self, x):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self):
        super(GaussianDistribution, self).__init__()

    @abstractmethod
    def entropy(self):
        pass

    @abstractmethod
    def neg_log_prob(self):
        pass

    @abstractmethod
    def sample(self, x):
        pass


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, num_classes):
        super(CategoricalDistribution, self).__init__()
        self.num_classes = num_classes

    def entropy(self, x):
        return x

    def neg_log_prob(self, x):
        return x

    def sample(self, p, use_logits=False):
        # todo utilize tf.distributions
        # a = tf.distributions.multinomail
        if use_logits:
            p = tf.math.softmax(p, axis=-1)
        actions = np.zeros(p.shape[0])
        for i in range(p.shape[0]):
            probs = p[i]
            actions[i] = np.random.choice(self.num_classes, p=p[i])
        return actions
