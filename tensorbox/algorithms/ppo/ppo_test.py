import unittest
import tensorflow as tf
from tensorflow.python import keras
from tensorbox.algorithms.ppo.trainer import PPOTrainer
from tensorbox.networks.shared_mlp import SharedMLPNet
import tensorbox.common.utils as utils

from tensorbox.common.vecenv import VecEnv


class TestPPO(unittest.TestCase):
    def test_trajectories(self):
        env_name = 'CartPole-v0'
        num_envs = 2
        # env = utils.make_env(env_name)
        env = VecEnv(env_name, num_envs=num_envs)
        horizon = 64
        # net = MLPNet(out_dim=5)
        net = SharedMLPNet(out_dims=(env.action_space.n, 1))
        opt = keras.optimizers.Adam()
        trainer = PPOTrainer(net=net,
                             opt=opt,
                             env=env,
                             horizon=horizon,
                             log_path='/var/tmp/delete_me')
        traj = trainer.get_trajectories()

        # adv = trainer.get_dataset(traj)
        # print('adv.shape=', adv.shape)
        self.assertTrue(traj.actions.shape == (horizon, num_envs),
                        'Output shape does not match input shape.')

    def test_get_dataset(self):
        env_name = 'CartPole-v0'
        num_envs = 2
        # env = utils.make_env(env_name)
        env = VecEnv(env_name, num_envs=num_envs)
        horizon = 64
        # net = MLPNet(out_dim=5)
        net = SharedMLPNet(out_dims=(env.action_space.n, 1))
        opt = keras.optimizers.Adam()
        trainer = PPOTrainer(net=net,
                             opt=opt,
                             env=env,
                             log_path='/var/tmp/delete_me')

        traj = trainer.get_trajectories()
        print(traj.actions.shape)
        ds = trainer.get_dataset(traj)
        self.assertTrue(isinstance(ds, tf.data.Dataset))
        # trainer.build_ppo_loss(ds)
        # for n_batch, batch in enumerate(ds):
        #     obs, acs, adv, tret = batch
        #     print('Batch n=', n_batch)

    # def test_ppo_loss(self):
    #     env_name = 'CartPole-v0'
    #     num_envs = 2
    #     # env = utils.make_env(env_name)
    #     env = VecEnv(env_name, num_envs=num_envs)
    #     horizon = 64
    #     # net = MLPNet(out_dim=5)
    #     net = SharedMLPNet(out_dims=(env.action_space.n, 1))
    #     opt = keras.optimizers.Adam()
    #     trainer = PPOTrainer(net=net,
    #                          opt=opt,
    #                          env=env,
    #                          log_path='/var/tmp/delete_me')
    #
    #     traj = trainer.get_trajectories()
    #     print(traj.actions.shape)
    #     ds = trainer.get_dataset(traj)
    #     print(ds)
    #     trainer.build_ppo_loss(ds)
    #     # for n_batch, batch in enumerate(ds):
    #     #     obs, acs, adv, tret, = batch
    #     #     print('Batch n=', n_batch)


if __name__ == '__main__':
    unittest.main()


