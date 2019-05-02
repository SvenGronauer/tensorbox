registered_functions = dict()


def register_function(function_name):
    """ register dataset functions into global dict"""

    def wrapper(func):
        registered_functions[function_name] = func
        return func
    return wrapper


@register_function('mnist')
def main_mnist(args):
    from .run_mnist import run_mnist
    run_mnist(args)


@register_function('unet')
def main_unet(args):
    from .run_unet_on_tetris import main
    main(args)
