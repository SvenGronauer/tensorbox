import numpy as np


def calculate_gae_advantages(trajectory, gamma=.99, kappa=0.95):
    """
    generalized advantage estimation (GAE) according to
    Schulman et. al - High-dimensional continuous control using generalized advantage estimation (2016)

    :param trajectory: Trajectory object, holding rewards, values, states etc.
    :param gamma: float, discount factor
    :param kappa: float, GAE-factor
    :return:
    """
    horizon, num_envs = trajectory.dones.shape
    rewards = trajectory.rewards
    dones = np.concatenate((trajectory.dones, np.zeros((1, num_envs))), axis=0)
    values_t_plus_1 = trajectory.values  # np.array, holding values of V(s_t) for t=0,..., T
    advantages = np.zeros([horizon, num_envs], dtype=np.float32)
    lastgaelam = 0.0

    for t in reversed(range(horizon)):
        non_terminal = 1. - dones[t+1]
        delta = rewards[t] + gamma * values_t_plus_1[t + 1] * non_terminal - values_t_plus_1[t]
        lastgaelam = delta + gamma * kappa * non_terminal * lastgaelam
        advantages[t] = lastgaelam

    return advantages


def calculate_target_returns(trajectory, gamma):
    """
    Calculate the discounted sum of rewards to train the value function as baseline

    :param trajectory:
    :param gamma:
    :return:
    """
    dones = trajectory.dones
    rewards = trajectory.rewards

    R_t = (1. - dones[-1]) * trajectory.values[-1] + dones[-1]

    horizon, num_envs = trajectory.dones.shape
    target_returns = np.zeros((horizon, num_envs), dtype=np.float32)

    for t in reversed(range(horizon)):
        R_t = (1.0 - dones[t]) * R_t * gamma + rewards[t]
        target_returns[t] = R_t

    return target_returns
