import unittest
import numpy as np
from vtrace import calculate_v_trace


def v_trace_google_numpy(discounts, log_rhos, rewards, values, bootstrap_value, clip_rho_threshold):
    """
    Calculates the ground truth for V-trace in Python/Numpy.

    Source: https://github.com/deepmind/scalable_agent/blob/master/vtrace_test.py

    This is a very inefficient way to calculate the V-trace ground truth.
    We calculate it this way because it is close to the mathematical notation of
    V-trace.
    v_s = V(x_s)
           + \sum^{T-1}_{t=s} \gamma^{t-s}
            * \prod_{i=s}^{t-1} c_i
             * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    Note that when we take the product over c_i, we write `s:t` as the notation
    of the paper is inclusive of the `t-1`, but Python is exclusive.
    Also note that np.prod([]) == 1.

    Note that arrays are of shape (horizon, batch_size)
    """
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)

    values_t_plus_1 = np.concatenate([values, bootstrap_value], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t], axis=0) * clipped_rhos[t] *
                    (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)

    return vs


def v_trace_numpy(rhos, rewards, values, gamma, bootstrap_value, clip_rho_threshold, clip_c_threshold):
    """
    calculate the ground truth for V-trace in numpy

    Note that arrays are of shape (batch_size, horizon)
    """
    
    assert clip_rho_threshold >= clip_c_threshold
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    cs = np.minimum(clipped_rhos, clip_c_threshold)
    batch_size, horizon = rewards.shape
    
    discounts = np.full(shape=horizon, fill_value=gamma)
    values_plus_bootstrap = np.concatenate([values, bootstrap_value], axis=1)

    vs = np.copy(values)
    for s in range(horizon):
        v_s = np.zeros(batch_size)
        for t in range(s, horizon):
            discount_prod = np.prod(discounts[s:t])
            cs_prod = np.prod(cs[:, s:t], axis=1)  # build product along axis 1 (horizon axis)
            rho_t = clipped_rhos[:, t]
            delta = rho_t * (rewards[:, t] + gamma * values_plus_bootstrap[:, t + 1] - values[:, t])
            v_s += (discount_prod * cs_prod * delta)
        vs[:, s] += v_s

    # calculate policy advantages
    vs_plus_bootstrap = np.concatenate([vs[:, 1:], bootstrap_value], axis=1)
    q_targets = clipped_rhos * (rewards + gamma * vs_plus_bootstrap - values)

    return vs, q_targets


class TestVTrace(unittest.TestCase):

    def test_v_trace(self):
        """ test V-trace, compare recursive vs forward calculation"""
        discount_factor = 0.95
        batch_size = 4
        horizon = 5

        clip_rho_threshold = 4.0
        clip_c_threshold = 1.0   

        policy_action_probs = np.stack([np.linspace(start=0.6, stop=0.9, num=horizon) for _ in range(batch_size)])
        behavior_action_probs = np.stack([np.linspace(start=0.3, stop=1.5, num=horizon) for _ in range(batch_size)])

        rhos = np.divide(policy_action_probs, behavior_action_probs)
        rewards = np.stack([np.linspace(start=0.0, stop=1.0, num=horizon) for _ in range(batch_size)])
        values = np.stack([np.linspace(start=0.0, stop=1.0, num=horizon) for _ in range(batch_size)])
        bootstrap_value = np.expand_dims(np.arange(0.0, batch_size) + 1.0, axis=-1)  # make shape (batch_size, 1)

        assert bootstrap_value.shape == (4, 1)
        
        discounts = np.full(shape=(horizon, batch_size), fill_value=discount_factor)
        google_ground_truth = v_trace_google_numpy(discounts=discounts,
                                                   log_rhos=np.log(np.swapaxes(rhos, 0, 1)),
                                                   rewards=np.swapaxes(rewards, 0, 1),
                                                   values=np.swapaxes(values, 0, 1),
                                                   bootstrap_value=np.swapaxes(bootstrap_value, 0, 1),
                                                   clip_rho_threshold=clip_rho_threshold)
        google_ground_truth = np.swapaxes(google_ground_truth, 0, 1)

        # recursive calculation
        recursive_vs, q_targets = calculate_v_trace(policy_action_probs=policy_action_probs,
                                                    values=values,
                                                    bootstrap_value=bootstrap_value,
                                                    rewards=rewards,
                                                    behavior_action_probs=behavior_action_probs,
                                                    rho_bar=clip_rho_threshold,
                                                    c_bar=clip_c_threshold,
                                                    gamma=discount_factor)
        # get ground truths by forward calculation
        gt_vs_targets, gt_q_targets = v_trace_numpy(rhos=rhos,
                                                    rewards=rewards,
                                                    values=values,
                                                    gamma=discount_factor,
                                                    bootstrap_value=bootstrap_value,
                                                    clip_rho_threshold=clip_rho_threshold,
                                                    clip_c_threshold=clip_c_threshold)

        self.assertTrue(np.allclose(gt_vs_targets, google_ground_truth))
        self.assertTrue(np.allclose(gt_vs_targets, recursive_vs))
        self.assertTrue(np.allclose(gt_q_targets, q_targets))


if __name__ == '__main__':
    unittest.main()
