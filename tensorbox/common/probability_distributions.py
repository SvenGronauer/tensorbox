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
    def __init__(self, shape, std):
        super(GaussianDistribution, self).__init__()
        self.shape = shape
        self.std_max = 0.5
        self.std_min = 0.15
        self.std = std
        self.dim = shape[0]  # number of dimension in action space

    def entropy(self, *args, **kwargs):
        ln = tf.math.log
        entropy = 0.5 * self.dim * (ln(2*np.pi) + 1) + self.dim * ln(self.std)
        return entropy

    def kl_divergence(self, p_as_logits, q_as_logits):
        assert p_as_logits.shape == q_as_logits.shape, 'shapes do not match'
        ln = tf.math.log
        p = p_as_logits
        q = q_as_logits
        # return tf.reduce_sum(ln(q.std) - ln(self.std) + (
        #         tf.square(self.std) + tf.square(self.mean - q.mean)) / (
        #                              2.0 * tf.square(q.std)) - 0.5, axis=-1)
        return tf.reduce_sum((tf.square(self.std) + tf.square(p - q)) / (2.0 * tf.square(self.std)) - 0.5,
                             axis=-1)

    def neg_log(self, actions, mean_as_logits, **kwargs):
        """ calculate -ln(p(.))
        :param actions: actions of behavior policy in [-1, 1.]
        :param mean_as_logits: logits of current policy
        :param kwargs:
        :return:
        """
        ln = np.log
        diff = actions - mean_as_logits
        p = tf.reduce_sum(tf.square(diff / self.std), axis=-1)
        res = 0.5 * p + 0.5 * self.dim * ln(2*np.pi) + ln(self.std) * self.dim
        return res

    def sample(self, logits, as_np_array=False):
        """ sample from gaussian distribution according to self.std and mean
        :param logits: tf.Tensor
        :param as_np_array: bool, if True returns np.array as result
        :return: tf.Tensor or np.array
        """
        ac = tf.random.normal(shape=logits.shape,
                              mean=logits,
                              stddev=self.std)
        if as_np_array:
            ac = ac.numpy()
        return ac

    def update(self, percentage_of_total):
        self.std = self.std_max - (self.std_max - self.std_min) * percentage_of_total


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, num_classes):
        super(CategoricalDistribution, self).__init__()
        self.num_classes = num_classes

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def entropy(self, p, from_logits=False, **kwargs):
        ln = tf.math.log
        if from_logits:
            p = tf.nn.softmax(p, axis=-1)
        return -1. * tf.reduce_sum(p * ln(p), axis=-1)

    # def get_action(self, p):
    #     action = self.sample(p).numpy().flatten()
    #     return action
    #
    # def get_sampled_action(self, p):
    #     """ same as get_action for CategoricalDistribution"""
    #     return self.get_action(p)

    def neg_log(self, action, p):
        """ implemented with sparse idx and p must be logits"""
        return self.sparse_neg_log(action, p, from_logits=True, axis=-1)

    def sample(self, logits, as_np_array=False):
        ac = tf.random.categorical(logits, 1)
        if as_np_array:
            ac = np.squeeze(ac.numpy()).flatten()
        return ac

    @staticmethod
    def sparse_neg_log(idx, p, from_logits, axis):
        return tf.losses.sparse_categorical_crossentropy(idx, p, from_logits, axis=axis)

    def update(self, percentage_of_total):
        pass  # nothing to do for categorial distribution
