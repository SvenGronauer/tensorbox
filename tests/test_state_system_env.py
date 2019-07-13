import unittest
import tensorflow as tf
from tensorflow.python import keras
from tensorbox.algorithms.ppo.trainer import PPOTrainer
from tensorbox.networks.shared_mlp import SharedMLPNet
import tensorbox.common.utils as utils

from tensorbox.common.vecenv import VecEnv
import numpy as np
from tensorbox.envs.control_systems import StateModel


class TestStateSystem(unittest.TestCase):
    def test_state_system_jump(self):
        R = 100  # resistance in Ohm
        C = 1.0e-3  # capacity in H
        T = 1 / (R * C)
        x_init = np.array([0.])
        A = np.array([-T])
        b = np.array([T])
        c = np.array([1.])
        d = np.array([0.])
        dt = 0.01  # equals 10 ms time steps
        model = StateModel(x_init, A, b, c, d, dt)

        ts = np.arange(0, 2, dt)
        ys = np.zeros_like(ts)
        ws = (np.arange(0, 2, dt) > 0.3).astype(np.float32)

        for i in range(len(ts)):
            ys[i] = model(ws[i])

        self.assertTrue(np.isclose(ys[-1], 1.),
                        'No jump of system')

        # plt.plot(ts, ws, 'b', ts, ys, 'r')
        # plt.show()


if __name__ == '__main__':
    unittest.main()


