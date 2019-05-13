

def get_default_parameters():
    return {
        'discount_factor': 0.99,
        'horizon': 1024,
        'learning_rate': 3.0e-4,
        'K': 1,
        'max_epochs': 500,
        'num_envs': 16
    }


def print_params(parsed_args,
                 learning_rate,
                 K,
                 horizon,
                 discount_factor,
                 schedule,
                 max_epochs,
                 num_envs,
                 log_directory,
                 *args,
                 **kwargs):
    """
    :param parsed_args: parsed arguments from main function
    :param horizon: int, number of roll outs in episode
    :param K: int, update cycle of old policy, defined as K in Schulman et al., 2017
    :param learning_rate: float, step size of gradient optimizer
    :param schedule: str, 'decay' for annealing clipping rate or 'const'
    :param max_epochs: int, maximum of performed update epochs
    :param create_checkpoints: bool
    :param restore_model: bool
    :param save_interval: int, save model parameters every n steps
    :param num_envs: int, number of available CPU cores
    :param log_directory: string, path where log files are written to
    :param settings: dict, configuration parameters for setup, parsed to sub-functions
    :return:
    """
    print('==================================')
    print('Parsed Parameters for PPO Training')
    print('================================== \n')
    print('Learning Rate:  {}'.format(learning_rate))
    print('K:              {}'.format(K))
    print('Horizon:        {}'.format(horizon))
    print('Discount:       {}'.format(discount_factor))
    print('Schedule:       {}'.format(schedule))
    print('Maximum Epochs: {}'.format(max_epochs))
    print('Cores:          {}'.format(num_envs))
    print('Restore Model:  {}'.format(restore_model))
    print('Create ckpts:   {}'.format(create_checkpoints))
    print('Save Interval:  {}'.format(save_interval))
    print('Run ID:         {}'.format(run_id))
    print('Log directory:  {}'.format(log_directory))
    print('\n==============================')
    args = parsed_args
    print('Parsed args: {}'.format(parsed_args))
