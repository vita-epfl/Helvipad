import argparse
import os

import numpy as np
import torch

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from models.base_model import apply_color_map
from models._360_igev_stereo._360_igev_model import _360IGEVModel
import general.stereo_datasets as datasets
import general.utils as utils

# Set the precision for matrix multiplication to medium for better performance
torch.set_float32_matmul_precision('medium')


def infer(cfg: DictConfig):
    """
    Perform inference using the trained model.

    Parameters:
    - cfg (DictConfig): Configuration object containing all inference parameters.
    """
    # Load the model checkpoint
    if cfg.restore_ckpt is not None:
        model_state = torch.load(cfg.restore_ckpt, weights_only=False)
    else:
        raise ValueError("No checkpoint provided for inference")

    # Create the directory to save inference results
    save_dir = os.path.join('models', '_360_igev_stereo', 'inference_results', cfg.infer_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the model in test mode
    model = _360IGEVModel(cfg, model_state, mode='test')
    model.model.eval()

    # Load the appropriate dataset for inference
    if cfg.dataset == 'helvipad':
        infer_set = datasets.HelvipadInference(cfg)
    elif cfg.dataset == '360SD':
        infer_set = datasets.RealWorld360SDInference(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    # Create a DataLoader for the inference dataset
    infer_loader = DataLoader(infer_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Perform inference on each batch
    for batch_idx, batch in enumerate(tqdm(infer_loader, desc="Inference Progress")):
        # Extract bottom and top images from the batch
        image_bottom, image_top = batch[1:3]

        # Perform model evaluation and get predictions
        if cfg.dataset == 'helvipad':
            depth_pred, disp_pred_deg, depth_gt, disp_gt_deg = model.evaluation_step(
                batch, batch_idx=batch_idx, mode='infer', log_images=False, log_image_step=0
            )
        elif cfg.dataset == '360SD':
            depth_pred, disp_pred_deg = model.evaluation_step(
                batch, batch_idx=batch_idx, mode='infer', log_images=False, log_image_step=0
            )
            depth_gt, disp_gt_deg = None, None

        # Convert images and predictions to NumPy arrays
        image_bottom_np = image_bottom[0].cpu().numpy().transpose(1, 2, 0)
        image_top_np = image_top[0].cpu().numpy().transpose(1, 2, 0)
        height = image_bottom_np.shape[1]
        depth_pred_np = depth_pred[0, 0].cpu().numpy()
        disp_pred_deg_np = disp_pred_deg[0, 0].cpu().numpy()
        if depth_gt is not None:
            depth_gt_np = depth_gt[0, 0].cpu().numpy()
        if disp_gt_deg is not None:
            disp_gt_deg_np = disp_gt_deg[0, 0].cpu().numpy()

        # Apply color maps to depth and disparity predictions
        depth_min, depth_max = depth_pred_np.min(), depth_pred_np.max()
        depth_pred_colored = apply_color_map(
            depth_pred_np, np.ones_like(depth_pred_np, dtype=bool), depth_min, depth_max, height
        )
        disp_min, disp_max = disp_pred_deg_np.min(), disp_pred_deg_np.max()
        disp_pred_colored = apply_color_map(
            disp_pred_deg_np, np.ones_like(disp_pred_deg_np, dtype=bool), disp_min, disp_max, height
        )

        # Save predictions and images
        name = cfg.images[batch_idx]
        np.save(os.path.join(save_dir, name + '_depth.npy'), depth_pred_np)
        np.save(os.path.join(save_dir, name + '_disp.npy'), disp_pred_deg_np)
        if depth_gt is not None:
            np.save(os.path.join(save_dir, name + '_depth_gt.npy'), depth_gt_np)
        if disp_gt_deg is not None:
            np.save(os.path.join(save_dir, name + '_disp_gt.npy'), disp_gt_deg_np)
        Image.fromarray(image_bottom_np.astype(np.uint8)).save(os.path.join(save_dir, name + '_bottom.png'))
        Image.fromarray(image_top_np.astype(np.uint8)).save(os.path.join(save_dir, name + '_top.png'))
        Image.fromarray(depth_pred_colored.astype(np.uint8)).save(os.path.join(save_dir, name + '_depth.png'))
        Image.fromarray(disp_pred_colored.astype(np.uint8)).save(os.path.join(save_dir, name + '_disp.png'))


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Modify Hydra configuration parameters.")

    # Define categorized argument options
    str_args = [
        'infer_name', 'restore_ckpt', 'dataset_root', 'model', 'dataset'
    ]
    int_args = [
        'val_circular_pad_size', 'max_disp'
    ]
    float_args = [
        'min_disp_deg', 'max_disp_deg'
    ]
    list_args = {
        'images': str
    }

    # Add arguments to the parser
    for arg in str_args:
        parser.add_argument(f'--{arg}', type=str, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg in int_args:
        parser.add_argument(f'--{arg}', type=int, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg in float_args:
        parser.add_argument(f'--{arg}', type=float, help=f'{arg.replace("_", " ").capitalize()}', default=None)
    for arg, typ in list_args.items():
        parser.add_argument(f'--{arg}', type=typ, nargs='+', help=f'{arg.replace("_", " ").capitalize()}', default=None)

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

    # Start the inference process
    infer(cfg)
