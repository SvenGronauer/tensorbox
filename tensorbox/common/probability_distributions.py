from abc import ABC, abstractmethod

import gym
import numpy as np
import tensorflow as tf


def get_probability_distribution(space):
    assert isinstance(space, gym.spaces.Space)
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalDistribution(num_classes=space.n)
    elif isinstance(space, gym.spaces.Box):
        return GaussianDistribution(shape=space.shape, std=0.5)
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

    def log_prob(self, action, p):
        return -self.neg_log(action, p)

    @abstractmethod
    def neg_log(self, action, p):
        pass

    @abstractmethod
    def sample(self, x):
        pass

    @abstractmethod
    def update(self, percentage_of_total):
        pass


class GaussianDistribution(ProbabilityDistribution):
    """ we assume each dimension to be independent """
    def __init__(self, shape, std, range=1.):
        super(GaussianDistribution, self).__init__()
        self.shape = shape
        self.std_max = 0.5
        self.std_min = 0.2
        self.std = std
        self.dim = shape[0]
        self.range = range
        self.activation = tf.nn.tanh

    def entropy(self, p, from_logits=False):
        ln = tf.math.log
        entropy = 0.5 * self.dim * (ln(2*np.pi) + 1) + self.dim * ln(self.std)
        return entropy

    def get_action(self, p_as_logits):
        """ returns mean of gaussian """
        return np.squeeze(np.clip(self.activation(p_as_logits).numpy(), -self.range, self.range))

    def get_sampled_action(self, p_as_logits):
        ac = self.sample(p_as_logits)
        return np.clip(ac.numpy(), -self.range, self.range)

    def kl_divergence(self, p_as_logits, q_as_logits):
        assert p_as_logits.shape == q_as_logits.shape, 'shapes do not match'
        ln = np.log
        p = self.activation(p_as_logits)
        q = self.activation(q_as_logits)
        # return tf.reduce_sum(ln(q.std) - ln(self.std) + (
        #         tf.square(self.std) + tf.square(self.mean - q.mean)) / (
        #                              2.0 * tf.square(q.std)) - 0.5, axis=-1)
        return tf.reduce_sum((tf.square(self.std) + tf.square(p - q)) / ( 2.0 * tf.square(self.std)) - 0.5,
                             axis=-1)

    def neg_log(self, actions, mean_as_logits, **kwargs):
        """ calculate -ln(p(.))
        :param actions: actions of behavior policy in [-1, 1.]
        :param mean_as_logits: logits of current policy
        :param kwargs:
        :return:
        """
        ln = np.log
        diff = actions - self.activation(mean_as_logits)
        p = tf.reduce_sum(tf.square(diff / self.std), axis=-1)
        res = 0.5 * p + 0.5 * self.dim * ln(2*np.pi) + ln(self.std) * self.dim
        return res

    def sample(self, prob_as_logits):
        """ sample from gaussian distribution according to self.std and mean
        :param prob_as_logits:
        :return: tf.Tensor,
        """
        mean = self.activation(prob_as_logits)
        ac = tf.random.normal(shape=self.shape,
                              mean=mean,
                              stddev=self.std)
        return tf.clip_by_value(ac, -self.range, self.range)

    def update(self, percentage_of_total):
        self.std = self.std_max - (self.std_max - self.std_min) * percentage_of_total


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

    def get_sampled_action(self, p):
        """ same as get_action for CategoricalDistribution"""
        return self.get_action(p)

    def neg_log(self, action, p):
        """ implemented with sparse idx and p must be logits"""
        return self.sparse_neg_log(action, p, from_logits=True, axis=-1)

    def sample(self, p_as_logits, **kwargs):
        return tf.random.categorical(p_as_logits, 1)

    @staticmethod
    def sparse_neg_log(idx, p, from_logits, axis):
        return tf.losses.sparse_categorical_crossentropy(idx, p, from_logits, axis=axis)

    def update(self, percentage_of_total):
        pass  # nothing to do for categorial distribution
