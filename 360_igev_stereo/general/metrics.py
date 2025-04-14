import warnings
import torch


class Metrics:
    """
    A class to compute various error metrics for depth estimation tasks.
    """
    def __init__(self):
        pass

    def rmse(self, gt, pred):
        """
        Compute the Root Mean Squared Error (RMSE).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - RMSE value.
        """
        mse = torch.mean((gt - pred) ** 2)
        return torch.sqrt(mse)

    def mae(self, gt, pred):
        """
        Compute the Mean Absolute Error (MAE).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - MAE value.
        """
        return torch.mean(torch.abs(gt - pred))

    def mare(self, gt, pred):
        """
        Compute the Mean Absolute Relative Error (MARE).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - MARE value.
        """
        return torch.mean(torch.abs((gt - pred) / gt))

    def a1(self, gt, pred):
        """
        Compute the Accuracy under threshold 1.25 (A1).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - A1 accuracy value.
        """
        thresh = torch.maximum((gt / pred), (pred / gt))
        return (thresh < 1.25).float().mean()

    def a2(self, gt, pred):
        """
        Compute the Accuracy under threshold 1.25^2 (A2).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - A2 accuracy value.
        """
        thresh = torch.maximum((gt / pred), (pred / gt))
        return (thresh < 1.25 ** 2).float().mean()

    def a3(self, gt, pred):
        """
        Compute the Accuracy under threshold 1.25^3 (A3).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - A3 accuracy value.
        """
        thresh = torch.maximum((gt / pred), (pred / gt))
        return (thresh < 1.25 ** 3).float().mean()

    def rmse_log(self, gt, pred):
        """
        Compute the Root Mean Squared Logarithmic Error (RMSE Log).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - RMSE Log value.
        """
        return torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    def log10(self, gt, pred):
        """
        Compute the Mean Log10 Error.

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - Log10 error value.
        """
        return (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()

    def sq_rel(self, gt, pred):
        """
        Compute the Squared Relative Error (SqRel).

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.

        Returns:
        - SqRel value.
        """
        return torch.mean(((gt - pred) ** 2) / gt)

    def compute_errors(self, gt, pred, valid, prefix=None):
        """
        Compute multiple error metrics between ground truth and predicted depth maps.

        Parameters:
        - gt: Ground truth tensor.
        - pred: Predicted tensor.
        - valid: Validity mask tensor.
        - prefix: Optional prefix for metric names.

        Returns:
        - Dictionary of computed error metrics.
        """
        valid = valid.unsqueeze(1).bool()
        gt_flat = gt[valid].flatten()
        pred_flat = pred[valid].flatten()

        if prefix:
            extended_prefix = f"{prefix}_"
        else:
            extended_prefix = ""

        errors = {
            f"{extended_prefix}rmse": self.rmse(gt_flat, pred_flat).item(),
            f"{extended_prefix}mae": self.mae(gt_flat, pred_flat).item(),
            f"{extended_prefix}mare": self.mare(gt_flat, pred_flat).item(),
            f"{extended_prefix}a1": self.a1(gt_flat, pred_flat).item(),
            f"{extended_prefix}a2": self.a2(gt_flat, pred_flat).item(),
            f"{extended_prefix}a3": self.a3(gt_flat, pred_flat).item(),
            f"{extended_prefix}rmse_log": self.rmse_log(gt_flat, pred_flat).item(),
            f"{extended_prefix}log10": self.log10(gt_flat, pred_flat).item(),
            f"{extended_prefix}sq_rel": self.sq_rel(gt_flat, pred_flat).item()
        }

        return errors


def calculate_lrce(depths_pred: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Left-Right Consistency Error (LRCE).

    Parameters:
    - depths_pred: Predicted depth tensor.
    - depth_gt: Ground truth depth tensor.
    - mask: Validity mask tensor.

    Returns:
    - LRCE value.
    """
    mask = mask.unsqueeze(1)

    # Extract first and last columns for gradient computation
    pred_first_col = depths_pred[:, :, :, 0]
    pred_last_col = depths_pred[:, :, :, -1]
    gt_first_col = depth_gt[:, :, :, 0]
    gt_last_col = depth_gt[:, :, :, -1]

    # Compute horizontal gradients for the entire tensor
    grad_pred = pred_first_col - pred_last_col
    grad_gt = gt_first_col - gt_last_col

    valid_mask_first = mask[:, :, :, 0]  # Mask for the first column
    valid_mask_last = mask[:, :, :, -1]  # Mask for the last column

    grad_pred = grad_pred * valid_mask_first.float() * valid_mask_last.float()
    grad_gt = grad_gt * valid_mask_first.float() * valid_mask_last.float()

    # Calculate LRCE: Ignore invalid points (they have a difference of 0)
    lrce_error = torch.abs(grad_gt - grad_pred)
    valid_points = valid_mask_first & valid_mask_last  # Combined mask for valid rows
    if valid_points.sum() == 0:  # If no valid points, return 0.0
        warnings.warn("No valid points found in the mask. Returning 0.0.")
        return torch.tensor(0.0)

    lrce = torch.mean(lrce_error[valid_points])

    return lrce.item()