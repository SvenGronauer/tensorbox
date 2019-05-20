
registered_datasets = dict()


def register_dataset(dataset_name):
    """ register dataset functions into global dict"""

    def wrapper(func):
        registered_datasets[dataset_name] = func
        return func
    return wrapper


@register_dataset("abc")
def get_abc_dataset():
    pass


@register_dataset('boston_housing')
def get_boston_housing(train_val_split):
    from tensorbox.datasets.boston_housing import create_boston_dataset
    return create_boston_dataset(train_val_split)


@register_dataset('cifar-10')
def get_cifar_10(train_val_split):
    from tensorbox.datasets.cifar_10 import create_cifar_10_dataset
    return create_cifar_10_dataset(train_val_split)


@register_dataset('lissajous')
def get_lissajous(train_val_split):
    from tensorbox.datasets.lissajous import create_lissajous_dataset
    return create_lissajous_dataset(train_val_split)


@register_dataset('mnist')
def get_mnist(train_val_split):
    from tensorbox.datasets.mnist import create_mnist_dataset
    return create_mnist_dataset(train_val_split)


@register_dataset('tetris')
def get_tetris_dataset(train_val_split):
    from tensorbox.datasets.segmentation_tetris import create_tetris_dataset
    return create_tetris_dataset(train_val_split)


def get_dataset(dataset_name, train_val_split=0.8, debug_level=0):
    """
    Get dataset according to dataset name

    :param dataset_name:
    :param train_val_split:
    :param debug_level:
    :return: tuple, consisting of train set and validation set
    """
    if debug_level >= 1:
        print('The registered datasets are:')
        print(registered_datasets)

    if dataset_name in registered_datasets:
        datasets = registered_datasets[dataset_name](train_val_split)
        return datasets
    else:
        raise KeyError('not a valid dataset.')