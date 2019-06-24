import unittest
import tensorflow as tf
from tensorflow.python import keras
from tensorbox.algorithms.ppo.trainer import PPOTrainer
from tensorbox.networks.shared_mlp import SharedMLPNet
import tensorbox.common.utils as utils

from tensorbox.common.vecenv import VecEnv


class TestDQN(unittest.TestCase):
    def test_replay_buffer(self):
        env_name = 'CartPole-v0'
        self.assertTrue(True, 'some condition is true.')

    def cost_value_iteration(self):
        env_name = 'CartPole-v0'
        self.assertTrue(True, 'some condition is true.')


if __name__ == '__main__':
    unittest.main()


