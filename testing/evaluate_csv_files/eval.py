import json
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class Configurations(object):
    def __init__(self):
        self._configs = list()
        self.counts = list()
        self.directories = list()
        self.current_idx = 0

    def append(self, config, directory):
        """ add configuration and directory to internal list"""
        try:
            idx = self._configs.index(config)
            self.counts[idx] += 1
            self.directories[idx].append(directory)
        except ValueError:  # add new config to list and increment the quantity to 1
            self._configs.append(config)
            self.counts.append(1)
            self.directories.append([directory])

    def __len__(self):
        return len(self._configs)

    def __next__(self):
        """ iterate over (config, directories) tuples """
        if self.current_idx < len(self):
            conf = self._configs[self.current_idx]
            dirs = self.directories[self.current_idx]
            self.current_idx += 1
            return conf, dirs
        else:
            self.current_idx = 0
            raise StopIteration

    def __iter__(self):
        """ iterate over (config, directories) tuples """
        return self


def get_label_from_config(config, values_of_interest):
    label = ''
    for n, values in enumerate(values_of_interest):
        keys = values.split('.')
        value = config
        for key in keys:
            value = value[key]
        label += str(value) + ' '
    return label


def generate_plots(root_directory,
                   csv_columns_of_interest,
                   config_values_of_interest,
                   config_file='config.json',
                   csv_file='summary.csv'):
    """
    automatically generate plots
    :param root_directory:
    :param csv_columns_of_interest:
    :param config_values_of_interest:
    :param config_file:
    :param csv_file:
    :return:
    """

    configurations = Configurations()
    # load configuration files into Configuration() class
    for directory in glob.glob(os.path.join(root_directory, '*')):
        with open(os.path.join(directory, config_file)) as json_file:
            config = json.load(json_file)
        configurations.append(config, directory)

    # iterate of configurations and plot values of interest
    for column in csv_columns_of_interest:
        fig, ax = plt.subplots()
        for config, directories in configurations:

            values = []
            for directory in directories:
                table = pd.read_csv(os.path.join(directory, csv_file))
                values.append(table[column].values)

            xs = table['global_step'].values
            values = np.array(values)
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)

            label_name = get_label_from_config(config, config_values_of_interest)
            a = 5
            plt.plot(xs, mean, label=label_name)
            plt.fill_between(xs, mean - std, mean + std, alpha=0.3)
        plt.title(column)
        plt.xlabel('Step')
        plt.ylabel(column)
        ax.set_yscale('log')
        plt.legend()
    plt.show()


def main():

    root_directory = '/var/tmp/ga87zej/compare_first_order_to_second_order'
    csv_columns_of_interest = ['loss_train', 'loss_test', 'time']

    config_values_of_interest = ['method.name', 'network.activation', 'network.units']
    generate_plots(root_directory, csv_columns_of_interest, config_values_of_interest)


if __name__ == '__main__':
    # args = utils.get_default_args()
    main()

