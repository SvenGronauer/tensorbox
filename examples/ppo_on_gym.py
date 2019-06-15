import tensorflow as tf
import numpy as np
from tensorbox.algorithms.ppo.trainer import PPOTrainer
from tensorbox.networks.shared_mlp import SharedMLPNet
import tensorbox.common.utils as utils

from tensorbox.common.vecenv import VecEnv
from tensorbox.common.probability_distributions import get_probability_distribution


def evaluate(args):
    env = utils.make_env(args.env)
    opt = tf.keras.optimizers.Adam()

    in_dim = env.observation_space.shape
    out_dims = env.action_space.shape + (1,)  # create tuple for shared network
    net = SharedMLPNet(in_dim=in_dim, out_dims=out_dims)

    trainer = PPOTrainer(net=net,
                         opt=opt,
                         env=env,
                         log_path='/var/tmp/delete_me')
    trainer.restore()

    done = False
    obs = env.reset()
    print('obs.shape=', obs.shape)
    ret = 0
    policy = get_probability_distribution(env.action_space)
    step = 0
    while not done:
        env.render()
        obs = np.expand_dims(obs, axis=0)
        # shaped_obs = obs.reshape(tuple([-1, ] + list(obs.shape)))
        ac_logits, val = trainer.net(obs)
        action = policy.get_action(ac_logits)
        obs, rew, done, _ = env.step(action)
        print('action:', action)
        ret += rew
        step += 1
    print('Episode return = {} after {} steps'.format(ret, step))
    env.close()


def run(args):
    env = VecEnv(args.env, num_envs=args.cores)
    opt = tf.keras.optimizers.Adam(lr=3.0e-4)

    in_dim = env.observation_space.shape
    out_dims = env.get_action_shape() + (1, )  # create tuple for shared network
    net = SharedMLPNet(in_dim=in_dim,
                       out_dims=out_dims,
                       activation=tf.nn.tanh)

    trainer = PPOTrainer(net=net,
                         opt=opt,
                         env=env,
                         log_dir='/var/tmp/delete_me')
    # trainer.restore()
    trainer.train(epochs=150)
    trainer.save()
    env.close()


if __name__ == '__main__':
    args = utils.get_default_args()
    args.env = 'RoboschoolReacher-v1'
    args.env = 'RoboschoolHumanoid-v1'
    args.env = 'CartPole-v1'
    # args.env = 'Pendulum-v0'
    # args.env = 'MountainCarContinuous-v0'
    print(args)
    run(args)
    # evaluate(args)


