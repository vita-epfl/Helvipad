import os
import argparse

from hydra import initialize, compose
from models._360_igev_stereo._360_igev_model import _360IGEVModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
import torch

import general.stereo_datasets as datasets
import general.utils as utils

# Set the precision for matrix multiplication to medium for better performance
torch.set_float32_matmul_precision('medium')


def train(cfg: DictConfig):
    """
    Main training function.

    Parameters:
    - cfg (DictConfig): Configuration object containing all training parameters.
    """
    # Initialize variables for tracking the best model
    best_mae = np.inf  # Best Mean Absolute Error (MAE) observed during validation
    best_model_state = None  # State dictionary of the best model
    filename = None  # Path to save the best model
    ckpt_path = cfg.restore_ckpt  # Path to restore a checkpoint if provided

    # Load model state from checkpoint if available and not in debug mode
    if ckpt_path is not None:
        model_state = torch.load(ckpt_path, weights_only=False)
    else:
        model_state = None

    # Initialize Weights & Biases (wandb) for experiment tracking if not in debug mode
    if not cfg.debug:
        config_wandb = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="360-IGEV-Stereo", name=cfg.exp_name, config=config_wandb)

    # Prepare datasets and data loaders for training and validation
    train_set = datasets.Helvipad(cfg, mode='train', kind='disparity', sequence=cfg.use_sequence)
    val_set = datasets.Helvipad(cfg, mode='val', kind='both', sequence=cfg.use_sequence)
    train_loader = DataLoader(train_set, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Calculate the total number of training steps and log image steps for validation
    num_steps = len(train_loader) * cfg.epochs
    log_image_steps_val = [int(round((len(val_loader) - 1) / (cfg.image_logs_per_eval - 1) * i)) for i in range(cfg.image_logs_per_eval)]

    # Initialize the model
    model = _360IGEVModel(cfg, model_state, num_steps)

    # Training loop
    for epoch in range(cfg.epochs):
        model.epoch = epoch  # Update the current epoch in the model

        # Training phase
        model.model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{cfg.epochs}")):
            model.training_step(batch, batch_idx)

        # Validation phase
        outputs = []
        log_image_step = -1
        model.model.eval()
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{cfg.epochs}")):
            log_images = batch_idx in log_image_steps_val
            if log_images:
                log_image_step += 1
            outputs.append(model.validation_step(batch, batch_idx, log_images, log_image_step))
        log_image_step += 1
        mae = model.validation_epoch_end(outputs, log_image_step)

        # Save the best model based on validation MAE
        if not cfg.debug:
            model.model.train()
            model.model.module.freeze_bn()  # Freeze batch normalization layers
            if mae < best_mae:
                if filename:
                    os.remove(filename)  # Remove the previous best model file
                filename = f"models/_360_igev_stereo/pretrained_models/{cfg.model_name}_{cfg.exp_name}_{epoch:02d}_{mae:.2f}.pth"
                best_model_state = model.model.state_dict()
                torch.save(best_model_state, filename)
                best_mae = mae

    # Testing phase
    test_set = datasets.Helvipad(cfg, mode='test', kind='both', sequence=cfg.use_sequence)
    test_loader = DataLoader(test_set, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    log_image_steps_test = [int(round((len(test_loader) - 1) / (cfg.image_logs_per_eval - 1) * i)) for i in range(cfg.image_logs_per_eval)]

    if not cfg.debug:
        model.update_model_state(best_model_state)  # Load the best model state for testing

    outputs = []
    model.model.eval()
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        log_images = batch_idx in log_image_steps_test
        if log_images:
            log_image_step += 1
        outputs.append(model.test_step(batch, batch_idx, log_images, log_image_step))
    log_image_step += 1
    model.test_epoch_end(outputs, log_image_step)

    # Finish the wandb session if not in debug mode
    if not cfg.debug:
        wandb.finish()


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Modify Hydra configuration parameters.")

    # Define all argument options
    bool_args = [
        'debug', 'augmented_gt', 'data_augmentation', 'do_photo'
    ]
    str_args = [
        'exp_name', 'dataset_root', 'model', 'restore_ckpt',
        'use_sequence', 'dataset'
    ]
    int_args = [
        'train_batch_size', 'val_batch_size', 'epochs',
        'max_disp', 'train_iters', 'valid_iters',
        'val_circular_pad_size', 'seed', 'image_logs_per_epoch_train', 'image_logs_per_eval'
    ]
    float_args = [
        'lr', 'min_disp_deg', 'max_disp_deg', 'wdecay'
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

    # Start the training process
    train(cfg)
