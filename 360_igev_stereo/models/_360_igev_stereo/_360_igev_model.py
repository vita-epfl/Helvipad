import os
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

from models.base_model import BaseModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../general')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.igev_stereo import IGEVStereo
from conversion import calculate_depth_map, disp_pix_to_disp_deg


class _360IGEVModel(BaseModel):
    """
    360IGEVModel class for training, validation, and inference of the 360-IGEV-Stereo model.
    Inherits from BaseModel.
    """
    def __init__(self, cfg, model_state, num_steps=1, mode='train'):
        """
        Initialize the 360IGEVModel.

        Parameters:
        - cfg: Configuration object.
        - model_state: Optional model state dictionary for loading.
        - num_steps: Total number of training steps.
        - mode: Mode of operation ('train', 'val', or 'test').
        """
        super().__init__(cfg, model_state, num_steps, mode=mode)

    def build_model(self):
        """
        Build the 360-IGEV-Stereo model and wrap it in DataParallel for multi-GPU support.
        """
        self.model = torch.nn.DataParallel(IGEVStereo(self.cfg), device_ids=[0])

    def update_model_state(self, model_state):
        """
        Update the model state with a new state dictionary.

        Parameters:
        - model_state: State dictionary to load into the model.
        """
        self.model.load_state_dict(model_state)

    def sequence_loss(self, pred, gt, valid):
        """
        Compute the sequence loss for disparity predictions.

        Parameters:
        - pred: Tuple containing disparity predictions and initial disparity prediction.
        - gt: Ground truth disparity.
        - valid: Valid mask.

        Returns:
        - disp_loss: Computed sequence loss.
        """
        disp_preds, disp_init_pred = pred
        disp_gt = gt
        loss_gamma = 0.9
        n_predictions = len(disp_preds)
        disp_loss = 0.0
        valid = valid.unsqueeze(1).bool()

        # Compute loss for the initial disparity prediction
        disp_loss += F.smooth_l1_loss(disp_init_pred[valid], disp_gt[valid], reduction='mean')

        # Compute loss for subsequent disparity predictions
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()[valid].mean()
            disp_loss += i_weight * i_loss

        return disp_loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Parameters:
        - batch: Training batch data.
        - batch_idx: Index of the current batch.
        """
        self.model.module.freeze_bn()  # Freeze batch normalization layers

        # Extract input data and move to GPU
        image_bottom, image_top, disp_gt_pix, valid = [x.cuda() for x in batch[1:]]

        # Forward pass through the model
        disp_init_pred, disp_preds = self.model(image_bottom, image_top, iters=self.cfg.train_iters)

        # Compute loss
        loss = self.sequence_loss((disp_preds, disp_init_pred), disp_gt_pix, valid)

        # Backpropagation and optimization
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Log training metrics
        self.log_training_step(loss, disp_preds[-1], disp_gt_pix, valid, image_bottom, 'disparity')
        self.step += 1

    def configure_optimizer(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        # Initialize AdamW optimizer
        params = self.model.parameters()
        self.optimizer = optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.optimizer.zero_grad()

        # Initialize OneCycleLR scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, self.cfg.lr, (self.num_steps + 100),
            pct_start=self.cfg.pct_start, cycle_momentum=False, anneal_strategy='linear'
        )

    def evaluation_step(self, batch, batch_idx, mode, log_images, log_image_step):
        """
        Perform a single evaluation step.

        Parameters:
        - batch: Evaluation batch data.
        - batch_idx: Index of the current batch.
        - mode: Mode of operation ('val' or 'test').
        - log_images: Whether to log images.
        - log_image_step: Step offset for logging images.

        Returns:
        - eval_results: Dictionary of evaluation results.
        """
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():  # Disable gradient calculation
            # Extract input data based on configuration
            if self.cfg.calc_lrce:
                image_bottom, image_top, depth_gt, disp_gt_pix, valid, depth_gt_aug, disp_gt_pix_aug, valid_gt_aug = [x.cuda() for x in batch[1:]]
            else:
                if mode == 'infer' and self.cfg.dataset == '360SD':
                    image_bottom, image_top = [x.cuda() for x in batch[1:]]
                else:
                    image_bottom, image_top, depth_gt, disp_gt_pix, valid = [x.cuda() for x in batch[1:]]

            # Add circular padding if configured
            if self.cfg.val_circular_pad_size:
                image_bottom = self.add_padding(image_bottom)
                image_top = self.add_padding(image_top)

            # Forward pass through the model
            disp_init_pred, disp_preds = self.model(image_bottom, image_top, iters=self.cfg.train_iters)
            disp_pred_pix = disp_preds[-1]

            # Remove padding if configured
            if self.cfg.val_circular_pad_size:
                disp_init_pred = self.remove_padding(disp_init_pred)
                disp_preds = [self.remove_padding(disp) for disp in disp_preds]

            # Compute loss if not in inference mode
            if mode != 'infer':
                loss = self.sequence_loss((disp_preds, disp_init_pred), disp_gt_pix, valid)

            # Remove padding from final predictions
            if self.cfg.val_circular_pad_size:
                disp_pred_pix = self.remove_padding(disp_pred_pix)

            # Compute depth predictions
            depth_pred = calculate_depth_map(disp_pred_pix, dataset=self.dataset)

            # Handle inference mode
            if mode == 'infer':
                disp_pred_deg = disp_pix_to_disp_deg(disp_pred_pix, dataset=self.dataset)
                if self.cfg.dataset == '360SD':
                    return depth_pred, disp_pred_deg
                else:
                    disp_gt_deg = disp_pix_to_disp_deg(disp_gt_pix, dataset=self.dataset)
                    return depth_pred, disp_pred_deg, depth_gt, disp_gt_deg

            # Evaluate metrics
            if self.cfg.calc_lrce:
                eval_results = self.evaluate_metrics(
                    image_bottom, depth_pred, depth_gt, disp_pred_pix, disp_gt_pix, valid, mode, loss,
                    log_images, log_image_step, depth_gt_aug, disp_gt_pix_aug, valid_gt_aug
                )
            else:
                eval_results = self.evaluate_metrics(
                    image_bottom, depth_pred, depth_gt, disp_pred_pix, disp_gt_pix, valid, mode, loss,
                    log_images, log_image_step
                )

        torch.backends.cudnn.benchmark = False
        return eval_results
