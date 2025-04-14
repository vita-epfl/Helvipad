import argparse

import torch
from tqdm import tqdm
import wandb

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models._360_igev_stereo._360_igev_model import _360IGEVModel
import general.stereo_datasets as datasets
import general.utils as utils

# Set the precision for matrix multiplication to medium for better performance
torch.set_float32_matmul_precision('medium')


def evaluate(cfg: DictConfig):
    """
    Perform evaluation using the trained model.

    Parameters:
    - cfg (DictConfig): Configuration object containing all evaluation parameters.
    """
    # Load the model checkpoint
    if cfg.restore_ckpt is not None:
        model_state = torch.load(cfg.restore_ckpt, weights_only=False)
    else:
        raise ValueError("No checkpoint provided for evaluation")

    # Initialize Weights & Biases (wandb) for experiment tracking if not in debug mode
    if not cfg.debug:
        config_wandb = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="360-IGEV-Stereo", name=cfg.exp_name, config=config_wandb)

    # Initialize the model in test mode
    model = _360IGEVModel(cfg, model_state, mode='test')

    # Load the test dataset
    test_set = datasets.Helvipad(cfg, mode='test', kind='both', sequence=cfg.use_sequence)
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.val_batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )

    # Calculate the steps at which images will be logged
    log_image_steps_test = [
        int(round((len(test_loader) - 1) / (cfg.image_logs_per_eval - 1) * i)) 
        for i in range(cfg.image_logs_per_eval)
    ]

    # Initialize variables for evaluation
    outputs = []
    log_image_step = -1
    model.model.eval()  # Set the model to evaluation mode

    # Iterate through the test dataset
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluation Progress")):
        # Determine whether to log images for this batch
        log_images = batch_idx in log_image_steps_test
        if log_images:
            log_image_step += 1

        # Perform the test step and collect outputs
        outputs.append(model.test_step(batch, batch_idx, log_images, log_image_step))

    # Finalize the evaluation by processing all outputs
    log_image_step += 1
    model.test_epoch_end(outputs, log_image_step)

    # Finish the wandb session if not in debug mode
    if not cfg.debug:
        wandb.finish()


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Modify Hydra configuration parameters.")

    # Define categorized argument options
    bool_args = [
        'calc_lrce', 'debug'
    ]
    str_args = [
        'exp_name', 'restore_ckpt', 'dataset_root', 'model', 
        'use_sequence', 'dataset'
    ]
    int_args = [
        'val_circular_pad_size', 'val_batch_size', 'max_disp', 'image_logs_per_eval'
    ]
    float_args = [
        'min_disp_deg', 'max_disp_deg'
    ]
    list_args = {'img_size': int}

    # Add arguments to the parser
    for arg in bool_args:
        parser.add_argument(f'--{arg}', type=utils.str2bool, help=f'Enable or disable {arg.replace("_", " ")}', default=None)
    for arg in str_args:
        parser.add_argument(f'--{arg}', type=str, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg in int_args:
        parser.add_argument(f'--{arg}', type=int, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg in float_args:
        parser.add_argument(f'--{arg}', type=float, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg, typ in list_args.items():
        parser.add_argument(f'--{arg}', type=typ, nargs='+', help=f'{arg.replace("_", " ").capitalize()}', default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Initialize Hydra and compose the configuration
    with initialize(config_path="conf", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={args.model}"] if args.model else [])

    # Open the struct and update configuration
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.model)
    utils.update_config_from_args(cfg, args)

    # Set random seed for reproducibility
    utils.set_seed(cfg.get('seed'))
    
    # Start the evaluation process
    evaluate(cfg)
