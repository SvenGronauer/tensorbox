from collections import namedtuple


Trajectory = namedtuple('Trajectory', ['observations',
                                       'actions',
                                       'rewards',
                                       'dones',
                                       'values',
                                       'mean_episode_return',
                                       'horizon'])

Dataset = namedtuple('Dataset', ['train',
                                 'test',
                                 'mean',
                                 'std',
                                 'x_shape',
                                 'y_shape'])
