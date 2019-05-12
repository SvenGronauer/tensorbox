from collections import namedtuple


Trajectory = namedtuple('Trajectory', ['observations',
                                       'actions',
                                       'rewards',
                                       'dones',
                                       'values',
                                       'mean_episode_return',
                                       'horizon'])
