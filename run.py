import tensorflow as tf
import argparse


def main(args):
    from examples.run_mnist import run_mnist
    run_mnist(args)


if __name__ == '__main__':
    print(tf.__version__)
    args = None
    main(args)
