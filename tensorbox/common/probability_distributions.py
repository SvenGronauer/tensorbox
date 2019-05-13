from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import gym


def get_probability_distribution(space):
    assert isinstance(space, gym.spaces.Space)
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalDistribution(num_classes=space.n)
    elif isinstance(space, gym.spaces.Box):
        return GaussianDistribution(shape=space.shape, std=0.2)
    else:
        raise NotImplementedError


class ProbabilityDistribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def entropy(self, p, from_logits=False):
        pass

    def get_action(self, p):
        pass

    @abstractmethod
    def neg_log(self, action, p):
        pass

    @abstractmethod
    def sample(self, x):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, shape, std):
        super(GaussianDistribution, self).__init__()
        self.shape = shape
        self.std = std
        self.dim = shape[0]
        print('GaussianDistribution.shape =', shape)

    def entropy(self, p, from_logits=False):
        ln = tf.math.log
        entropy = 0.5 * self.dim * (ln(2*np.pi*self.std**2) + 1)
        return entropy

    def get_action(self, p):
        return p

    def neg_log(self, actions, mean, **kwargs):
        ln = tf.math.log
        p = tf.reduce_sum(tf.square((actions - mean) / self.std), axis=-1)
        res = 0.5 * p + ln(2*np.pi*self.std**2) * 0.5 * self.dim
        return res

    def sample(self, mean):
        n = tf.random.normal(shape=self.shape,
                             mean=mean,
                             stddev=self.std)
        return n


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, num_classes):
        super(CategoricalDistribution, self).__init__()
        self.num_classes = num_classes

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def entropy(self, p, from_logits=False, **kwargs):
        if from_logits:
            p = tf.nn.softmax(p, axis=-1)
        return -1. * tf.reduce_sum(p * tf.math.log(p), axis=-1)

    def get_action(self, p):
        action = self.sample(p).numpy()
        return int(np.squeeze(action))

    def neg_log(self, action, p):
        """ implemented with sparse idx and p must be logits"""
        return self.sparse_neg_log(action, p, from_logits=True, axis=-1)

    def sample(self, p_as_logits, **kwargs):
        return tf.random.categorical(p_as_logits, 1)

    @staticmethod
    def sparse_neg_log(idx, p, from_logits, axis):
        return tf.losses.sparse_categorical_crossentropy(idx, p, from_logits, axis=axis)

