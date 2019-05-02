import tensorflow as tf
import argparse
import datetime
from examples import registered_functions


def main(args):
    if args.debug >= 1:
        print(tf.__version__)
        print('Registered run functions:')
        [print('*\t', f) for f in registered_functions]

    if args.debug >= 2:
        print('Log Path =', args.log_path)

    if args.func in registered_functions:
        func = registered_functions[args.func]
        func(args)

    elif args.func == 'test':
        raise NotImplementedError
    else:
        print('Registered run functions:')
        [print('*\t', f) for f in registered_functions]
        print('Function {} is not defined!'.format(args.func))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                                 'Cartpole.')
    parser.add_argument('func', help='Function to be executed')
    parser.add_argument('--alg', dest='algorithm', default='ppo', type=str,
                        help='Choose between \'a3c\' and \'ppo\'.')
    parser.add_argument('--debug', default=0, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--env', default='singlecar', type=str,
                        help='Choose between \'fruit\',\'singlecar\', \'reacher\', and \'cart\'.')
    parser.add_argument('--log', dest='log_path', default='/var/tmp/ga87zej/',
                        help='Set the seed for random generator')
    parser.add_argument('--seed', dest='seed', default=None,
                        help='Set the seed for random generator')
    args = parser.parse_args()

    args.log_path += args.func + '/' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    main(args)
