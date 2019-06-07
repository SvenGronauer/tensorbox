"""
This file is used to register all defined data sets in the directory
of tensorbox.datasets

"""
registered_datasets = dict()


def register_dataset(dataset_name):
    """ register dataset functions into global dict"""
    def wrapper(func):
        registered_datasets[dataset_name] = func
        return func
    return wrapper


@register_dataset('boston_housing')
def get_boston_housing(train_val_split, **kwargs):
    from tensorbox.datasets.boston_housing import create_boston_dataset
    return create_boston_dataset(train_val_split)


@register_dataset('cifar-10')
def get_cifar_10(train_val_split, **kwargs):
    from tensorbox.datasets.cifar_10 import create_cifar_10_dataset
    return create_cifar_10_dataset(train_val_split)


@register_dataset('lissajous')
def get_lissajous(train_val_split, **kwargs):
    from tensorbox.datasets.lissajous import create_lissajous_dataset
    return create_lissajous_dataset()


@register_dataset('mnist')
def get_mnist(train_val_split, **kwargs):
    from tensorbox.datasets.mnist import create_mnist_dataset
    return create_mnist_dataset(train_val_split, **kwargs)


@register_dataset('tetris')
def get_tetris_dataset(train_val_split, **kwargs):
    from tensorbox.datasets.segmentation_tetris import create_tetris_dataset
    return create_tetris_dataset(train_val_split)


@register_dataset('unsupervised_gaussian')
def get_unsupervised_gaussian_dataset(train_val_split, **kwargs):
    from tensorbox.datasets.unsupervised_gaussian import create_unsupervised_gaussian_dataset
    return create_unsupervised_gaussian_dataset(normalize=False)


def get_dataset(dataset_name,
                train_val_split=0.8,
                debug_level=0,
                **kwargs):
    """ Get dataset according to dataset name

    kwargs:
        extra_mappings: additional functions that are applied to data set.

    :param dataset_name:
    :param train_val_split:
    :param debug_level:
    :param: tuple, holding functions that are mapped to TF dataset
    :return: tuple, consisting of train set and validation set
    """

    if debug_level >= 1:
        print('The registered datasets are:')
        print(registered_datasets)

    if dataset_name in registered_datasets:
        dataset = registered_datasets[dataset_name](train_val_split=train_val_split,
                                                    debug_level=debug_level,
                                                    **kwargs)
        return dataset
    else:
        raise KeyError('The data set name "{}" has not been registered'.format(dataset_name))
