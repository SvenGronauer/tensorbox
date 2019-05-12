import tensorflow as tf
import numpy as np
from tensorbox.algorithms.ppo.trainer import PPOTrainer
from tensorbox.networks.shared_mlp import SharedMLPNet
import tensorbox.common.utils as utils

from tensorbox.common.vecenv import VecEnv
from tensorbox.common.probability_distributions import get_probability_distribution


def evaluate(args):
    env = utils.make_env()
    opt = tf.keras.optimizers.Adam()

    in_dim = env.observation_space.shape
    net = SharedMLPNet(in_dim=in_dim, out_dims=(env.action_space.n, 1))

    trainer = PPOTrainer(net=net,
                         opt=opt,
                         env=env,
                         log_path='/var/tmp/delete_me')
    trainer.restore('/var/tmp/delete_me')

    done = False
    obs = env.reset()
    ret = 0
    policy = get_probability_distribution(env.action_space)
    while not done:
        # env.render()
        obs = obs.reshape((-1, 4))
        ac_logits, val = net(obs)
        action = policy(ac_logits, from_logits=True).numpy()
        print(np.squeeze(action))
        action = int(np.squeeze(action))
        obs, rew, done, _ = env.step(action)
        ret += rew
    print('Episode return =', ret)


def run(args):
    env_name = 'CartPole-v0'
    num_envs = 4
    env = VecEnv(env_name, num_envs=num_envs)
    opt = tf.keras.optimizers.Adam(lr=3.0e-4)

    in_dim = env.observation_space.shape
    net = SharedMLPNet(in_dim=in_dim,
                       out_dims=(env.action_space.n, 1),
                       activation=tf.nn.tanh)

    trainer = PPOTrainer(net=net,
                         opt=opt,
                         env=env,
                         log_path='/var/tmp/delete_me')
    trainer.restore('/var/tmp/delete_me')
    trainer.train(epochs=200)
    trainer.save()


if __name__ == '__main__':
    args = utils.get_default_args()
    print(args)
    run(args)
    # evaluate(args)


