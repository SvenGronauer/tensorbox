import tensorflow as tf
from tensorflow.python import keras

""" tensorbox imports"""
from tensorbox.networks import MLPNet
from tensorbox.datasets import get_dataset
from tensorbox.common import utils
from tensorbox.common.logger import CSVLogger
from tensorbox.common.classes import Configuration
from tensorbox.methods import LevenbergMarquardt, GradientDescent
from tensorbox.common.trainer import SupervisedTrainer

# set this flag to fix cudNN bug on RTX graphics card
tf.config.gpu.set_per_process_memory_growth(True)


def main(args, dataset_name, units, activation, use_marquardt=True, **kwargs):
    dataset = get_dataset(dataset_name)
    train_epochs = 100
    lr = 1.0e-3
    log_dir = args.log_dir if args.log_dir else '/var/tmp/ga87zej'
    logger = CSVLogger(log_dir, total_steps=train_epochs, stdout=False)
    net = MLPNet(in_dim=dataset.x_shape,
                 out_dim=dataset.y_shape,
                 activation=activation,
                 units=units)
    opt = tf.keras.optimizers.SGD(lr=lr) if use_marquardt else tf.keras.optimizers.Adam(lr=lr)

    loss_func = keras.losses.MeanSquaredError()
    loss_metric = keras.metrics.Mean(name='mean')

    method = LevenbergMarquardt(loss_func) if use_marquardt else GradientDescent(loss_func)

    config = Configuration(net=net,
                           opt=opt,
                           method=method,
                           dataset=dataset,
                           logger=logger,
                           log_dir=log_dir)
    config.dump()
    trainer = SupervisedTrainer(from_config=config,
                                loss_func=loss_func,
                                metric=loss_metric,
                                dataset=dataset)
    trainer.train(total_steps=train_epochs)


def param_search():
    list_units = [(8, 8), (16, 16), (32, 32)]
    list_activations = ['relu', 'tanh']
    modes = [True, False]
    dataset_names = ['boston_housing', 'lissajous']
    runs_per_setting = 5

    for ds_name in dataset_names:
        for units in list_units:
            for activation in list_activations:
                for use_marquardt in modes:
                    for i in range(runs_per_setting):
                        args = utils.get_default_args()
                        print('=' * 80)
                        string = 'Data set: {}, Units: {}, Activation: {}, Marquardt? {}'
                        print(string.format(ds_name, units, activation, use_marquardt))
                        main(args, dataset_name=ds_name, units=units,
                             activation=activation, use_marquardt=use_marquardt)


if __name__ == '__main__':
    param_search()
