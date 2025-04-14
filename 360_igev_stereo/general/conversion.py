import torch



def compute_depth_from_disparity(
    disparity_map: torch.Tensor, y_grid: torch.Tensor
) -> torch.Tensor:
    """
    Compute depth from disparity using trigonometric projection.

    The depth is computed based on the relationship:

        depth = B * (sin(theta) / tan(disparity_rad) + cos(theta))

    where:
        - B is the baseline (fixed at 0.191 meters for Helvipad).
        - theta is the vertical angle corresponding to each pixel in the y-grid.
        - disparity_rad is the disparity map scaled to radians.

    Parameters:
        disparity_map (torch.Tensor): Disparity values (bs, h, w).
        y_grid (torch.Tensor): Corresponding y-coordinates for each pixel.

    Returns:
        torch.Tensor: Computed depth values (bs, h, w).
    """
    B: float = 0.191  # Baseline distance (meters)
    height: int = 1920  # Original image height
    height_down: int = 960  # Downscaled height for disparity

    theta_grid = y_grid * torch.pi / height
    disparity_map_rad = (torch.pi / height_down) * disparity_map

    depth_map = (
        (torch.sin(theta_grid) / torch.tan(disparity_map_rad)) + torch.cos(theta_grid)
    ) * B

    return depth_map


def compute_depth_from_disparity_360sd(disparity_map: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute depth from disparity for the 360SD dataset using trigonometric projection.

    The depth is computed based on the relationship:

        depth = B * (sin(theta) / tan(disparity_rad) + cos(theta))

    where:
        - B is the baseline (fixed at 0.2 meters for 360SD).
        - theta is the vertical angle corresponding to each pixel in the y-grid.
        - disparity_rad is the disparity map scaled to radians.

    Parameters:
        disparity_map (torch.Tensor): Disparity values (bs, h, w).
        y_grid (torch.Tensor): Corresponding y-coordinates for each pixel.

    Returns:
        torch.Tensor: Computed depth values (bs, h, w).
    """
    B: float = 0.2  # Baseline distance (meters)
    height: int = 512  # Image height for 360SD dataset

    theta_grid = y_grid * torch.pi / height
    disparity_map_rad = disparity_map * (torch.pi / height)

    depth_map = B * ((torch.sin(theta_grid) / torch.tan(disparity_map_rad)) + torch.cos(theta_grid))

    return depth_map


def calculate_depth_map(disparity_map: torch.Tensor, dataset: str = 'helvipad') -> torch.Tensor:
    """
    Convert a disparity map to a depth map based on dataset-specific calibration.

    The conversion uses dataset-specific y-grid computations and applies the appropriate
    `compute_depth_from_disparity` function.

    Parameters:
        disparity_map (torch.Tensor): Input tensor of shape (bs, 1, h, w) or (bs, h, w).
        dataset (str): Dataset name ('helvipad' or '360SD').

    Returns:
        torch.Tensor: Depth map (bs, h, w).
    """
    # Check if the input has a channel dimension (bs, 1, h, w)
    has_channel_dim: bool = disparity_map.dim() == 4 and disparity_map.shape[1] == 1
    if has_channel_dim:
        disparity_map = disparity_map.squeeze(1)  # Remove the channel dimension

    # Initialize variables
    bs, height, width = disparity_map.shape
    non_zero_disparity = disparity_map != 0
    depth_map = torch.zeros_like(disparity_map, dtype=torch.float32)

    # Compute the y-grid based on the dataset
    if dataset == 'helvipad':
        y_grid = torch.arange(
            512 + 2 * height - 1,
            512,
            step=-2,
            device=disparity_map.device
        ).unsqueeze(0).unsqueeze(-1)
    elif dataset == '360SD':
        y_grid = torch.arange(
            511.5,
            0,
            step=-1,
            device=disparity_map.device
        ).unsqueeze(0).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    y_grid = y_grid.expand(bs, -1, width)

    # Filter non-zero disparity values and corresponding y-grid values
    filtered_disparity = disparity_map[non_zero_disparity]
    filtered_y_grid = y_grid[non_zero_disparity]

    # Compute depth based on the dataset
    if dataset == 'helvipad':
        depth_map[non_zero_disparity] = compute_depth_from_disparity(
            filtered_disparity, filtered_y_grid
        )
    elif dataset == '360SD':
        depth_map[non_zero_disparity] = compute_depth_from_disparity_360sd(
            filtered_disparity, filtered_y_grid
        )

    # Restore the channel dimension if the input had one
    if has_channel_dim:
        depth_map = depth_map.unsqueeze(1)  # Add back the channel dimension (bs, 1, h, w)

    return depth_map


def disp_deg_to_disp_pix(disp_deg, dataset='helvipad'):
    """
    Convert a disparity value from degrees to pixels.

    Parameters:
        disp_deg (float): Disparity in degrees.
        dataset (str): Dataset name ('helvipad', '360SD').

    Returns:
        float: Disparity in pixels.
    """
    if dataset == 'helvipad':
        H_down = 960
        disp_pix = (H_down / 180) * disp_deg
    elif dataset == '360SD':
        H = 512
        disp_pix = (H / 180) * disp_deg
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return disp_pix


def disp_pix_to_disp_deg(disp_pix, dataset='helvipad'):
    """
    Convert a disparity value from pixels to degrees.

    Parameters:
        disp_pix (float): Disparity in pixels.
        dataset (str): Dataset name ('helvipad', '360SD').

    Returns:
        float: Disparity in degrees.
    """
    if dataset == 'helvipad':
        H_down = 960
        disp_deg = (180 / H_down) * disp_pix
    elif dataset == '360SD':
        H = 512
        disp_deg = (180 / H) * disp_pix
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return disp_deg