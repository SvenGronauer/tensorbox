import numpy as np
from collections import namedtuple

VTrace = namedtuple('VTrace', ['v_s', 'policy_adv', 'clipped_rho'])


def calculate_v_trace(policy_action_probs,
                      values,
                      rewards,
                      bootstrap_value,
                      behavior_action_probs,
                      gamma=0.99,
                      rho_bar=1.0,
                      c_bar=1.0):
    """
    calculate V-trace targets for off-policy actor-critic learning recursively
    as proposed in: Espeholt et al. 2018, IMPALA

    :param policy_action_probs:
    :param values:
    :param rewards:
    :param bootstrap_value:
    :param behavior_action_probs:
    :param gamma:
    :param rho_bar:
    :param c_bar:
    :return: V-trace targets, shape=(batch_size, sequence_length)
    """
    assert values.ndim == 2, 'Please provide 2d-arrays of shape (batch_size, sequence_length)'
    assert rewards.ndim == 2
    assert policy_action_probs.ndim == 2
    assert behavior_action_probs.ndim == 2
    assert c_bar <= rho_bar

    batch_size, sequence_length = values.shape
    rhos = np.divide(policy_action_probs, behavior_action_probs)
    clip_rhos = np.minimum(rhos, rho_bar)
    clip_cs = np.minimum(rhos, c_bar)
    values_plus_bootstrap = np.concatenate([values, bootstrap_value], axis=1)

    v_s = np.copy(values)
    v_s_plus_1 = bootstrap_value[:, 0]  # bootstrap from last state

    # calculate v_s
    for t in reversed(range(sequence_length)):
        delta = clip_rhos[:, t] * (rewards[:, t] + gamma * values_plus_bootstrap[:, t+1] - values[:, t])
        v_s[:, t] += delta + gamma * clip_cs[:, t] * (v_s_plus_1 - values_plus_bootstrap[:, t+1])
        v_s_plus_1 = v_s[:, t]  # accumulate current v_s for next iteration

    # calculate q_targets
    v_s_plus_1 = np.concatenate([v_s[:, 1:], bootstrap_value], axis=1)
    policy_advantage = clip_rhos * (rewards + gamma * v_s_plus_1 - values)

    return VTrace(v_s=v_s,
                  policy_adv=policy_advantage,
                  clipped_rho=clip_rhos)
    # return v_s, policy_advantage
