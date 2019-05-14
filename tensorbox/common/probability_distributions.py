from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import gym


def get_probability_distribution(space):
    assert isinstance(space, gym.spaces.Space)
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalDistribution(num_classes=space.n)
    elif isinstance(space, gym.spaces.Box):
        return GaussianDistribution(shape=space.shape, std=0.15)
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
    """ we assume each dimension to be independent """
    def __init__(self, shape, std):
        super(GaussianDistribution, self).__init__()
        self.shape = shape
        self.std = std
        self.dim = shape[0]
        # print('GaussianDistribution.shape =', shape)

    def entropy(self, p, from_logits=False):
        ln = tf.math.log
        entropy = 0.5 * self.dim * (ln(2*np.pi) + 1) + self.dim * ln(self.std)
        return entropy

    def get_action(self, p_as_logits):
        # a = p_as_logits.numpy()
        ac = tf.nn.tanh(p_as_logits).numpy()
        if ac.shape[0] == 1:
            ac = np.squeeze(ac, axis=0)
        return ac

    def neg_log(self, actions, mean_as_logits, **kwargs):
        """ calculate -ln(p(.))
        :param actions: actions of behavior policy in [-1, 1.]
        :param mean_as_logits: logits of current policy
        :param kwargs:
        :return:
        """
        ln = tf.math.log
        act = tf.nn.tanh
        p = tf.reduce_sum(tf.square((actions - act(mean_as_logits)) / self.std), axis=-1)
        res = 0.5 * p + 0.5 * self.dim * ln(2*np.pi) + ln(self.std) * self.dim
        return res

    def sample(self, p_as_logits):
        mean = tf.nn.tanh(p_as_logits)
        ac = tf.random.normal(shape=self.shape,
                              mean=mean,
                              stddev=self.std)

        return tf.clip_by_value(ac, -1., 1.)


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
        action = self.sample(p).numpy().flatten()
        return action

    def neg_log(self, action, p):
        """ implemented with sparse idx and p must be logits"""
        return self.sparse_neg_log(action, p, from_logits=True, axis=-1)

    def sample(self, p_as_logits, **kwargs):
        return tf.random.categorical(p_as_logits, 1)

    @staticmethod
    def sparse_neg_log(idx, p, from_logits, axis):
        return tf.losses.sparse_categorical_crossentropy(idx, p, from_logits, axis=axis)

