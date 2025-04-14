import argparse
import random

import numpy as np
import torch


def str2bool(v):
    """
    Convert a string to a boolean value.

    Parameters:
    - v: Input string or boolean.

    Returns:
    - bool: Converted boolean value.

    Raises:
    - argparse.ArgumentTypeError: If the input cannot be interpreted as a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed_value):
    """
    Set the seed for reproducibility across various libraries.

    Parameters:
    - seed_value: Integer seed value to set. If None, no seed is set.
    """
    if seed_value is not None:
        # Set seed for PyTorch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        # Set seed for NumPy and Python's random module
        np.random.seed(seed_value)
        random.seed(seed_value)

        # Ensure deterministic behavior in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_config_from_args(cfg, args):
    """
    Update the configuration object using command-line arguments.

    Parameters:
    - cfg: Configuration object to update.
    - args: Parsed command-line arguments.

    Updates:
    - Attributes in the configuration object are updated with non-None values from args.
    """
    for arg, value in vars(args).items():
        if value is not None:
            setattr(cfg, arg, value)