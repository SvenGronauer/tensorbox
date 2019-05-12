from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import gym


def get_probability_distribution(space):
    assert isinstance(space, gym.spaces.Space)
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalDistribution(num_classes=space.n)
    elif isinstance(space, gym.spaces.Box):
        raise NotImplementedError
    else:
        raise NotImplementedError


class ProbabilityDistribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def entropy(self, p, from_logits=False):
        pass

    @abstractmethod
    def neg_log(self, x):
        pass

    @abstractmethod
    def sample(self, x):
        pass

    @abstractmethod
    def sparse_neg_log(self, idx, p):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self):
        super(GaussianDistribution, self).__init__()

    @abstractmethod
    def entropy(self, p, from_logits=False):
        raise NotImplementedError

    def neg_log(self, x):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, num_classes):
        super(CategoricalDistribution, self).__init__()
        self.num_classes = num_classes

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def entropy(self, p, from_logits=False):
        if from_logits:
            p = tf.nn.softmax(p, axis=-1)
        return -1. * tf.reduce_sum(p * tf.math.log(p), axis=-1)

    def neg_log(self, x):
        return x

    def sample(self, p_as_logits, **kwargs):
        return tf.random.categorical(p_as_logits, 1)

    def sparse_neg_log(self, idx, p, from_logits=False):
        return tf.losses.sparse_categorical_crossentropy(idx, p, from_logits, axis=-1)

