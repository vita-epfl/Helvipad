import os
import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../general')))
from conversion import calculate_depth_map, disp_pix_to_disp_deg
from metrics import Metrics, calculate_lrce


class BaseModel:
    def __init__(self, cfg, model_state, num_steps, mode):
        """
        Initialize the base model with configuration and optional states.

        Parameters:
        - cfg: Configuration object.
        - model_state: Optional model state dictionary for loading.
        - num_steps: Total number of training steps.
        - mode: Mode of operation ('train', 'val', or 'test').
        """
        self.cfg = cfg
        self.dataset = self.cfg.dataset
        self.num_steps = num_steps
        self.n_steps_per_epoch = self.num_steps // self.cfg.epochs
        self.example_ct = 0
        self.epoch = 0
        self.step = 0
        self.log_image_steps_train = [
            int(round((self.n_steps_per_epoch - 1) / (self.cfg.image_logs_per_epoch_train - 1) * i))
            for i in range(self.cfg.image_logs_per_epoch_train)
        ]
        self.build_model()
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=False)
        self.model.cuda()
        if mode == 'train':
            self.configure_optimizer()
            self.scaler = torch.GradScaler("cuda", enabled=self.cfg.mixed_precision)

    def build_model(self):
        """
        Build the model.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method for model building.")

    def configure_optimizer(self):
        """
        Configure the optimizers.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method for optimizer configuration.")

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        Should be implemented by subclasses.

        Parameters:
        - batch: The training batch data.
        - batch_idx: Index of the current batch.
        """
        raise NotImplementedError("Subclasses must implement this method for training.")

    def sequence_loss(self, pred, gt, valid):
        """
        Calculate the sequence loss.

        Parameters:
        - pred: Predicted depth.
        - gt: Ground truth depth.
        - valid: Valid mask.
        """
        raise NotImplementedError("Subclasses must implement this method for sequence loss calculation.")

    def log_training_step(self, loss, pred, gt, valid, image_bottom, kind):
        """
        Log training metrics and images to Weights & Biases.

        Parameters:
        - loss: Training loss.
        - pred: Predicted depth.
        - gt: Ground truth depth.
        - valid: Valid mask.
        - image_bottom: Input image.
        - kind: Type of training data.
        """
        if not self.cfg.debug:
            self.example_ct += len(image_bottom)
            metrics = {
                "train/train_loss": loss.item(),
                "train/epoch": (self.step + 1) / self.n_steps_per_epoch,
                "train/example_ct": self.example_ct,
                "train/lr": self.scheduler.get_last_lr()[0]
            }
            wandb.log(metrics, step=self.step)
            if (self.step - self.n_steps_per_epoch * self.epoch) in self.log_image_steps_train:
                if kind == "disparity":
                    depth_pred = calculate_depth_map(pred.detach(), dataset=self.dataset)
                    depth_gt = calculate_depth_map(gt.detach(), dataset=self.dataset)
                elif kind == "depth":
                    depth_pred = pred.detach()
                    depth_gt = gt.detach()
                self.log_images(image_bottom, depth_gt, depth_pred, valid, mode="train")

    def validation_step(self, batch, batch_idx, log_images=False, log_image_step=0):
        """
        Perform a single validation step.

        Parameters:
        - batch: Validation batch data.
        - batch_idx: Index of the current batch.
        - log_images: Whether to log images.
        - log_image_step: Step offset for logging images.

        Returns:
        - Evaluation results for the batch.
        """
        eval_results = self.evaluation_step(batch, batch_idx, mode="val", log_images=log_images, log_image_step=log_image_step)
        return eval_results

    def evaluation_step(self, batch, batch_idx, mode, log_images, log_image_step):
        """
        Perform a single evaluation step.
        Should be implemented by subclasses.

        Parameters:
        - batch: Evaluation batch data.
        - batch_idx: Index of the current batch.
        - mode: Mode of operation ('val' or 'test').
        - log_images: Whether to log images.
        - log_image_step: Step offset for logging images.
        """
        raise NotImplementedError("Subclasses must implement this method for evaluation.")

    def add_padding(self, image):
        """
        Add circular padding to the image.

        Parameters:
        - image: Input image.

        Returns:
        - Padded image.
        """
        pad_size = self.cfg.val_circular_pad_size
        padding = (pad_size, pad_size, 0, 0)
        image = F.pad(image, padding, mode='circular')
        return image

    def remove_padding(self, pred):
        """
        Remove circular padding from the prediction.

        Parameters:
        - pred: Input prediction.

        Returns:
        - Prediction with padding removed.
        """
        pad_size = self.cfg.val_circular_pad_size
        pred = pred[:, :, :, pad_size:-pad_size]
        return pred

    def validation_epoch_end(self, outputs, step_offset):
        """
        Finalize validation by logging metrics.

        Parameters:
        - outputs: List of validation outputs.
        - step_offset: Step offset for logging.

        Returns:
        - Mean Absolute Error (MAE) for validation.
        """
        mae = self.log_epoch_end(outputs, "val", step_offset)
        return mae

    def evaluate_metrics(self, image_bottom, depth_pred, depth_gt, disp_pred_pix, disp_gt_pix, valid, mode, loss, log_images, log_image_step, depth_gt_aug=None, disp_gt_pix_aug=None, valid_gt_aug=None):
        """
        Evaluate metrics for the given predictions and ground truth.

        Parameters:
        - image_bottom: Input image.
        - depth_pred: Predicted depth.
        - depth_gt: Ground truth depth.
        - disp_pred_pix: Predicted disparity in pixels.
        - disp_gt_pix: Ground truth disparity in pixels.
        - valid: Valid mask.
        - mode: Mode of operation ('val' or 'test').
        - loss: Loss value.
        - log_images: Whether to log images.
        - log_image_step: Step offset for logging images.
        - depth_gt_aug: Augmented ground truth depth (optional).
        - disp_gt_pix_aug: Augmented ground truth disparity in pixels (optional).
        - valid_gt_aug: Augmented valid mask (optional).

        Returns:
        - Dictionary of evaluation results.
        """
        eval_results = {"loss": loss.item()}
        metrics = Metrics()
        assert depth_pred.shape == depth_gt.shape
        if valid.dim() == 4:
            valid = valid.squeeze(1)
        eval_results.update(metrics.compute_errors(depth_gt, depth_pred, valid))

        if mode == "test":
            disp_pred_deg = disp_pix_to_disp_deg(disp_pred_pix, dataset=self.dataset)
            disp_gt_deg = disp_pix_to_disp_deg(disp_gt_pix, dataset=self.dataset)
            assert disp_pred_deg.shape == disp_gt_deg.shape
            eval_results.update(metrics.compute_errors(disp_gt_deg, disp_pred_deg, valid, prefix="disp"))
            if hasattr(self.cfg, 'calc_lrce') and self.cfg.calc_lrce:
                eval_results['lrce'] = calculate_lrce(depth_pred, depth_gt_aug, valid_gt_aug)
                disp_gt_deg_aug = disp_pix_to_disp_deg(disp_gt_pix_aug, dataset=self.dataset)
                eval_results['disp_lrce'] = calculate_lrce(disp_pred_deg, disp_gt_deg_aug, valid_gt_aug)

        if log_images and not self.cfg.debug:
            if hasattr(self.cfg, 'val_circular_pad_size') and self.cfg.val_circular_pad_size:
                image_bottom = self.remove_padding(image_bottom)
            self.log_images(image_bottom, depth_gt, depth_pred, valid, mode, log_image_step)

        return eval_results

    def test_step(self, batch, batch_idx, log_images=False, log_image_step=0):
        """
        Perform a single test step.

        Parameters:
        - batch: Test batch data.
        - batch_idx: Index of the current batch.
        - log_images: Whether to log images.
        - log_image_step: Step offset for logging images.

        Returns:
        - Evaluation results for the batch.
        """
        eval_results = self.evaluation_step(batch, batch_idx, mode="test", log_images=log_images, log_image_step=log_image_step)
        return eval_results

    def test_epoch_end(self, outputs, step_offset):
        """
        Finalize testing by logging metrics.

        Parameters:
        - outputs: List of test outputs.
        - step_offset: Step offset for logging.
        """
        self.log_epoch_end(outputs, "test", step_offset)

    def log_images(self, image_bottom, depth_gt, depth_pred, valid, mode, log_image_step=0):
        """
        Log images to Weights & Biases.

        Parameters:
        - image_bottom: Input image.
        - depth_gt: Ground truth depth.
        - depth_pred: Predicted depth.
        - valid: Valid mask.
        - mode: Mode of operation ('train', 'val', or 'test').
        - log_image_step: Step offset for logging images.
        """
        # Convert tensors to numpy arrays
        image_np = image_bottom[0].cpu().numpy().transpose(1, 2, 0)
        depth_gt_np = depth_gt[0, 0].cpu().numpy()
        depth_pred_np = depth_pred[0, 0].cpu().numpy()
        valid_mask = valid[0].cpu().numpy().astype(bool)
        width = image_np.shape[1]

        # Calculate depth range for normalization
        depth_min = min(depth_gt_np[valid_mask].min(), depth_pred_np.min())
        depth_max = max(depth_gt_np.max(), depth_pred_np.max())

        # Apply color mapping to depth maps
        depth_gt_colored = apply_color_map(depth_gt_np, valid_mask, depth_min, depth_max, width)
        depth_pred_colored = apply_color_map(depth_pred_np, np.ones_like(depth_pred_np, dtype=bool), depth_min, depth_max, width)

        # Compute absolute and relative error maps
        abs_error_map = np.abs(depth_gt_np - depth_pred_np)
        abs_error_map[~valid_mask] = 0
        rel_error_map = abs_error_map / (depth_gt_np + 1e-6)
        rel_error_map[~valid_mask] = 0

        # Apply color mapping to error maps
        abs_error_colored = apply_color_map(abs_error_map, valid_mask, 0, abs_error_map.max(), width)
        rel_error_colored = apply_color_map(rel_error_map, valid_mask, 0, rel_error_map.max(), width)

        # Organize images for logging with the original keys
        images = {
            "image_bottom": image_np,
            "depth_gt": depth_gt_colored,
            "depth_pred": depth_pred_colored,
            "depth_ae": abs_error_colored,
            "depth_are": rel_error_colored
        }

        # Log images to Weights & Biases
        wandb_images = {mode: [wandb.Image(image, caption=key) for key, image in images.items()]}
        wandb.log(wandb_images, step=self.step + log_image_step)

    def log_epoch_end(self, outputs, mode, step_offset):
        """
        Log metrics at the end of an epoch.

        Parameters:
        - outputs: List of outputs from the epoch.
        - mode: Mode of operation ('train', 'val', or 'test').
        - step_offset: Step offset for logging metrics.

        Returns:
        - Mean Absolute Error (MAE) for validation mode.
        """
        keys = outputs[0].keys()
        prefix = f"{mode}/{mode}_"
        avg_metrics_dict = {}
        for key in keys:
            avg_metric = np.mean([x[key] for x in outputs])
            avg_metrics_dict.update({prefix + key: avg_metric})
        if not self.cfg.debug:
            wandb.log(avg_metrics_dict, step=self.step + step_offset)
        metrics_str = ', '.join([f"{key[len(prefix):]}: {value}" for key, value in avg_metrics_dict.items()])
        print(f"Metrics {mode} after epoch {self.epoch}: {metrics_str}")
        if mode == 'val':
            mae = avg_metrics_dict['val/val_mae']
            return mae

    def update_model_state(self, model_state):
        """
        Update the model state with a new state dictionary.

        Parameters:
        - model_state: State dictionary to load into the model.
        """
        self.model.load_state_dict(model_state)


def apply_color_map(data, mask, vmin, vmax, width, color_gradient='gist_rainbow'):
    """
    Apply a color map to the given data.

    Parameters:
    - data: Input data to color map.
    - mask: Valid mask for the data.
    - vmin: Minimum value for normalization.
    - vmax: Maximum value for normalization.
    - width: Width of the image for visualization.
    - color_gradient: Color gradient to use for the color map.

    Returns:
    - Color-mapped data with an optional color bar.
    """
    # Normalize the data
    normalized_data = np.clip((data - vmin) / (vmax - vmin + 1e-6), 0, 1) if vmin != vmax else np.zeros_like(data)

    # Apply color map
    colormap = matplotlib.cm.get_cmap(color_gradient)
    color_mapped_data = colormap(normalized_data, bytes=True)

    # Mark invalid regions as gray
    color_mapped_data[~mask] = (128, 128, 128, 255)

    # Create a figure and axis without a plot
    fig, ax = plt.subplots(figsize=(width / 100, 0.5), dpi=100)
    ax.axis('off')  # Turn off the axis

    # Adjust the position of the colorbar using bounding box
    bbox = ax.get_position()
    colorbar_height = 0.15  # Set the thickness of the colorbar here
    cax = plt.axes([0.5 - bbox.width / 2, 0.5 - colorbar_height / 2, bbox.width, colorbar_height])  # Centered colorbar

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Only needed for ScalarMappable
    plt.colorbar(sm, cax=cax, orientation='horizontal')

    # Save the colorbar to an image
    fig.canvas.draw()
    colorbar_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    colorbar_image = colorbar_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    # Convert color_mapped_data_resized to 3 channels (discarding alpha)
    color_mapped_data_rgb = color_mapped_data[:, :, :3]
    
    # Concatenate the RGB color_mapped_data and colorbar_image vertically
    color_mapped_with_bar = np.concatenate((color_mapped_data_rgb, colorbar_image), axis=0)

    return color_mapped_with_bar