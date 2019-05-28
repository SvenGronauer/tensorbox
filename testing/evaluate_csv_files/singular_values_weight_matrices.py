import json
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    root_directory = '/var/tmp/ga87zej/testing/2019_05_28__23_19_03/'

    # file_path = os.path.join(root_directory, 'singular_values.csv')
    csv_file = 'singular_values.csv'
    data = np.genfromtxt(os.path.join(root_directory, csv_file), delimiter=',')

    N = 2

    xs = np.arange(len(data) / N) * N
    data = data[::N].T  # transpose to evaluate column-wise and take only every N column

    # table = pd.read_csv(os.path.join(root_directory, csv_file), header=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    ax.set_title('Default violin plot')
    ax.set_ylabel('Observed values')
    ax.violinplot(data, positions=xs, showmeans=True)

    plt.show()
    a = 2


if __name__ == '__main__':
    # args = utils.get_default_args()
    main()

