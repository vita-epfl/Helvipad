import numpy as np
import torch
import torch.utils.data as data
import os
import random
import cv2
from glob import glob
from PIL import Image

from augmentor import SparseFlowAugmentor
from conversion import disp_deg_to_disp_pix


class StereoDataset(data.Dataset):
    def __init__(self, cfg, mode, kind):
        """
        Initializes the StereoDataset.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        - mode: Mode of operation ('train', 'val', 'test').
        - kind: Type of data to load ('disparity', 'both', 'no_gt').
        """
        # Determine dataset type based on the dataset root path
        if 'helvipad' in cfg.dataset_root:
            self.dataset = 'helvipad'
        elif 'data_360SD' in cfg.dataset_root:
            self.dataset = '360SD'
        else:
            raise ValueError("Unsupported dataset type in dataset_root.")

        self.kind = kind
        self.calc_lrce = getattr(cfg, 'calc_lrce', False)  # Check if LRCE calculation is enabled

        # Initialize data lists
        self.image_list = []
        self.disparity_list = []
        if self.calc_lrce:
            self.disparity_aug_list = []
        if self.kind == 'both':
            self.depth_list = []
            if self.calc_lrce:
                self.depth_aug_list = []

        # Initialize data augmentation if enabled
        data_augm = cfg.data_augmentation and mode == 'train'
        if data_augm:
            gamma = cfg.img_gamma or [1, 1, 1, 1]  # Default gamma values if not provided
            self.augmentor = SparseFlowAugmentor(cfg.do_photo, cfg.saturation_range, gamma)
        else:
            self.augmentor = None

    def __getitem__(self, index):
        """
        Retrieves a data sample by index.

        Parameters:
        - index: Index of the sample to retrieve.

        Returns:
        - Tuple containing images, depth, disparity, and other relevant data.
        """
        index = index % len(self.image_list)  # Handle index overflow

        # Load and preprocess the bottom and top images
        img_bottom = self.load_and_process_image(self.image_list[index][0])
        img_top = self.load_and_process_image(self.image_list[index][1])

        # Load depth and disparity based on the specified kind
        depth, disp = None, None
        depth_gt_aug, disp_gt_pix_aug, valid_gt_aug = None, None, None
        valid = None
        if self.kind == 'both':
            depth, valid = self.load_depth(index)
            if self.calc_lrce:
                depth_gt_aug, valid_gt_aug = self.load_depth(index, augmented=True)

        if self.kind in {'disparity', 'both'}:
            disp, valid = self.load_disparity(index)
            if self.calc_lrce:
                disp_gt_pix_aug, valid_gt_aug = self.load_disparity(index, augmented=True)

        # Apply data augmentation if enabled
        if self.augmentor is not None:
            img_bottom, img_top, depth, disp, valid = self.augmentor(img_bottom, img_top, depth, disp, valid)

        # Convert images and data to PyTorch tensors
        img_bottom = self._to_tensor(img_bottom)
        img_top = self._to_tensor(img_top)
        if depth is not None:
            depth = self._to_tensor(depth, remove_first_channel=True)
        if disp is not None:
            disp = self._to_tensor(disp, remove_first_channel=True)
        if self.calc_lrce:
            depth_gt_aug = self._to_tensor(depth_gt_aug, remove_first_channel=True)
            disp_gt_pix_aug = self._to_tensor(disp_gt_pix_aug, remove_first_channel=True)

        # Return data based on the kind
        return self._prepare_output(index, img_bottom, img_top, depth, disp, valid, depth_gt_aug, disp_gt_pix_aug, valid_gt_aug)

    def load_and_process_image(self, image_path):
        """
        Load an image from the specified path, convert it to RGB if required, and ensure it has 3 channels.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - np.ndarray: Processed image as a NumPy array with 3 channels.
        """
        try:
            # Open the image
            img = Image.open(image_path)

            # Convert to RGB only if the dataset is '360SD'
            if self.dataset == '360SD':
                img = img.convert('RGB')

            # Convert the image to a NumPy array and ensure it has 3 channels
            img = np.array(img).astype(np.uint8)
            if img.shape[-1] > 3:  # Handle images with more than 3 channels
                img = img[..., :3]

            return img

        except Exception as e:
            # Handle errors and provide meaningful feedback
            raise RuntimeError(f"Failed to load and process image at {image_path}: {e}")

    def load_depth(self, index, augmented=False):
        """Load and process the depth image."""
        if self.dataset == 'helvipad':
            depth = self.readDepthHelvipad(self.depth_aug_list[index] if augmented else self.depth_list[index])
        elif self.dataset == '360SD':
            depth = self.readDepth360SD(self.depth_list[index])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        depth, valid = depth
        depth = np.array(depth).astype(np.float32)
        depth = np.stack([np.zeros_like(depth), depth], axis=-1)
        return depth, valid

    def load_disparity(self, index, augmented=False):
        """Load and process the disparity image."""
        if self.dataset == 'helvipad':
            disp = self.readDisparityHelvipad(self.disparity_aug_list[index] if augmented else self.disparity_list[index])
        elif self.dataset == '360SD':
            disp = self.readDisparity360SD(self.disparity_list[index])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        disp, valid = disp
        disp = np.array(disp).astype(np.float32)
        disp = disp_deg_to_disp_pix(disp, self.dataset)
        disp = np.stack([np.zeros_like(disp), disp], axis=-1)
        return disp, valid

    def _to_tensor(self, array, remove_first_channel=False):
        """
        Convert a NumPy array to a PyTorch tensor.

        Parameters:
        - array (np.ndarray): Input array.
        - remove_first_channel (bool): Whether to remove the first channel.

        Returns:
        - torch.Tensor: Converted tensor.
        """
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        if remove_first_channel:
            tensor = tensor[1:]
        return tensor

    def _prepare_output(self, index, img_bottom, img_top, depth, disp, valid, depth_gt_aug, disp_gt_pix_aug, valid_gt_aug):
        """
        Prepare the output data based on the kind.

        Parameters:
        - index: Index of the sample.
        - img_bottom, img_top: Processed bottom and top images.
        - depth, disp: Depth and disparity data.
        - valid: Validity mask.
        - depth_gt_aug, disp_gt_pix_aug, valid_gt_aug: Augmented data.

        Returns:
        - Tuple containing the prepared output data.
        """
        if self.kind == 'disparity':
            return self.image_list[index] + [self.disparity_list[index]], img_bottom, img_top, disp, valid
        elif self.kind == 'both':
            if self.calc_lrce:
                return (
                    self.image_list[index] + [self.depth_list[index], self.disparity_list[index], self.depth_aug_list[index], self.disparity_aug_list[index]],
                    img_bottom, img_top, depth, disp, valid, depth_gt_aug, disp_gt_pix_aug, valid_gt_aug
                )
            else:
                return self.image_list[index] + [self.depth_list[index], self.disparity_list[index]], img_bottom, img_top, depth, disp, valid
        elif self.kind == 'no_gt':
            return self.image_list[index], img_bottom, img_top
        else:
            raise ValueError(f"Unknown kind {self.kind}")
        
    def readDepthHelvipad(self, filename):
        """
        input : dispartiy filed in half size
        disp shape : (H, W)
        valid shape : (H, W) fills with True or False
        """
        depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
        valid = depth > 0.0
        return depth, valid

    def readDisparityHelvipad(self, filename):
        """
        input : dispartiy filed in half size
        disp shape : (H, W)
        valid shape : (H, W) fills with True or False
        """
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 2048.0
        valid = disp > 0.0
        return disp, valid

    def readDisparity360SD(self, filename):
        """
        input : dispartiy filed in half size
        disp shape : (H, W)
        valid shape : (H, W) fills with True or False
        """
        disp = np.load(filename)
        valid = disp > 0.0
        return disp, valid

    def readDepth360SD(self, filename):
        """
        input : dispartiy filed in half size
        disp shape : (H, W)
        valid shape : (H, W) fills with True or False
        """
        depth = np.load(filename)
        valid = depth > 0.0
        return depth, valid

    def __len__(self):
        return len(self.image_list)


class Helvipad(StereoDataset):
    def __init__(self, cfg, mode, kind, sequence):
        """
        Initializes the Helvipad dataset.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        - mode: Mode of operation ('train', 'val', 'test').
        - kind: Type of data to load ('disparity', 'both').
        - sequence: Sequence type ('ALL', 'IN', 'OUT', 'INOUT', 'NOUT').
        """
        super(Helvipad, self).__init__(cfg, mode, kind)

        # Validate dataset root
        dataset_root = cfg.dataset_root
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        # Determine depth maps directory based on mode and augmentation settings
        depth_maps = self._get_depth_maps_directory(mode, cfg)

        # Get the mode directory (train/val/test)
        mode_directory = self._get_mode_directory(mode)

        # Get the list of depth files based on the sequence type
        depth_list = self._get_depth_list(dataset_root, mode_directory, depth_maps, sequence)

        # Apply debug or data reduction settings if applicable
        depth_list = self._apply_debug_or_reduction(cfg, depth_list, mode)

        # Generate corresponding file lists for images and disparities
        self.image_list = self._generate_image_list(depth_list, depth_maps, 'images_bottom', 'images_top')
        self.disparity_list = self._generate_file_list(depth_list, 'depth_maps', 'disparity_maps')

        # Generate augmented file lists if required
        if kind == 'both':
            self.depth_list = depth_list
            if self.calc_lrce:
                self.depth_aug_list = self._generate_file_list(depth_list, 'depth_maps', 'depth_maps_augmented')
                self.disparity_aug_list = self._generate_file_list(depth_list, 'depth_maps', 'disparity_maps_augmented')

        # Print dataset statistics
        print(f"Number of images for Helvipad ({kind}) in mode '{mode}' with sequences '{sequence}': {len(self.image_list)}")

    def _get_depth_maps_directory(self, mode, cfg):
        """
        Determines the depth maps directory based on mode and augmentation settings.

        Parameters:
        - mode: Mode of operation ('train', 'val', 'test').
        - cfg: Configuration object.

        Returns:
        - str: Depth maps directory name.
        """
        depth_maps = 'depth_maps'
        if mode == 'train' and cfg.augmented_gt:
            depth_maps += '_augmented'
        return depth_maps

    def _get_mode_directory(self, mode):
        """
        Returns the directory name corresponding to the mode.

        Parameters:
        - mode: Mode of operation ('train', 'val', 'test').

        Returns:
        - str: Directory name.
        """
        mode_mapping = {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        }
        if mode not in mode_mapping:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_mapping[mode]

    def _get_depth_list(self, dataset_root, mode_directory, depth_maps, sequence):
        """
        Retrieves the list of depth files based on the sequence type.

        Parameters:
        - dataset_root: Root directory of the dataset.
        - mode_directory: Directory corresponding to the mode (train/val/test).
        - depth_maps: Directory containing depth maps.
        - sequence: Sequence type ('ALL', 'IN', 'OUT', 'INOUT', 'NOUT').

        Returns:
        - List[str]: List of depth file paths.
        """
        sequence_mapping = {
            'ALL': os.path.join(dataset_root, mode_directory, depth_maps, '*', '*.png'),
            'IN': os.path.join(dataset_root, mode_directory, depth_maps, '*_IN', '*.png'),
            'OUT': os.path.join(dataset_root, mode_directory, depth_maps, '*_OUT', '*.png'),
            'INOUT': [
                os.path.join(dataset_root, mode_directory, depth_maps, '*_IN', '*.png'),
                os.path.join(dataset_root, mode_directory, depth_maps, '*_OUT', '*.png')
            ],
            'NOUT': os.path.join(dataset_root, mode_directory, depth_maps, '*_NOUT', '*.png')
        }

        if sequence not in sequence_mapping:
            raise ValueError(f"Unknown sequence type: {sequence}")

        if sequence == 'INOUT':
            in_list = sorted(glob(sequence_mapping['INOUT'][0], recursive=True))
            out_list = sorted(glob(sequence_mapping['INOUT'][1], recursive=True))
            return sorted(in_list + out_list)
        else:
            return sorted(glob(sequence_mapping[sequence], recursive=True))

    def _apply_debug_or_reduction(self, cfg, depth_list, mode):
        """
        Applies debug or data reduction settings to the depth list.

        Parameters:
        - cfg: Configuration object.
        - depth_list: List of depth file paths.
        - mode: Mode of operation ('train', 'val', 'test').

        Returns:
        - List[str]: Modified depth list.
        """
        if cfg.debug:
            return depth_list[::500]  # Use every 500th file for debugging
        elif mode == 'train' and hasattr(cfg, 'data_reduction_factor'):
            random_generator = random.Random(cfg.seed)
            reduction_factor = cfg.data_reduction_factor
            return random_generator.sample(depth_list, len(depth_list) // reduction_factor)
        return depth_list

    def _generate_file_list(self, base_list, old_dir, new_dir):
        """
        Generates a list of file paths by replacing directory names.

        Parameters:
        - base_list: List of base file paths.
        - old_dir: Directory name to replace.
        - new_dir: New directory name.

        Returns:
        - List[str]: List of updated file paths.
        """
        return [path.replace(old_dir, new_dir) for path in base_list]

    def _generate_image_list(self, base_list, depth_maps, bottom_dir, top_dir):
        """
        Generates a list of image file paths for bottom and top images.

        Parameters:
        - base_list: List of base file paths.
        - depth_maps: Directory containing depth maps.
        - bottom_dir: Directory name for bottom images.
        - top_dir: Directory name for top images.

        Returns:
        - List[List[str]]: List of [bottom, top] image file paths.
        """
        bottom_list = [path.replace(depth_maps, bottom_dir).replace('.png', '.jpg') for path in base_list]
        top_list = [path.replace(depth_maps, top_dir).replace('.png', '.jpg') for path in base_list]
        return [[bottom, top] for bottom, top in zip(bottom_list, top_list)]


class HelvipadInference(StereoDataset):
    def __init__(self, cfg):
        """
        Initializes the HelvipadInference dataset.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        """
        super().__init__(cfg, mode="inf", kind="both")

        # Validate dataset root
        dataset_root = cfg.dataset_root
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        # Generate file lists for images, depth maps, and disparity maps
        self.image_list, self.depth_list, self.disparity_list = self._generate_file_lists(cfg, dataset_root)

        # Print dataset statistics
        print(f"Number of images for HelvipadInference: {len(self.image_list)}")

    def _generate_file_lists(self, cfg, dataset_root):
        """
        Generates file lists for images, depth maps, and disparity maps.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        - dataset_root: Root directory of the dataset.

        Returns:
        - Tuple[List[List[str]], List[str], List[str]]: Image list, depth list, and disparity list.
        """
        image_top_list = []
        for img in cfg.images:
            try:
                # Parse the image identifier (mode-directory, sequence, index)
                mode_directory, sequence, index = img.split("-")
                # Generate the top image path
                image_top_list.append(os.path.join(dataset_root, mode_directory, 'images_top', sequence, f"{index}.jpg"))
            except ValueError:
                raise ValueError(f"Invalid image identifier format: {img}. Expected format: 'mode-sequence-index'.")

        # Generate corresponding bottom image paths
        image_bottom_list = [img.replace('top', 'bottom') for img in image_top_list]

        # Generate depth and disparity map paths
        depth_list = [img.replace('images_top', 'depth_maps').replace('.jpg', '.png') for img in image_top_list]
        disparity_list = [img.replace('images_top', 'disparity_maps').replace('.jpg', '.png') for img in image_top_list]

        # Combine bottom and top image paths into a single list
        image_list = [[bottom, top] for bottom, top in zip(image_bottom_list, image_top_list)]

        return image_list, depth_list, disparity_list


class RealWorld360SDInference(StereoDataset):
    def __init__(self, cfg):
        """
        Initializes the RealWorld360SDInference dataset.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        """
        super().__init__(cfg, mode="inf", kind="no_gt")

        # Validate dataset root
        dataset_root = cfg.dataset_root
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        # Generate file lists for images
        self.image_list = self._generate_image_list(cfg, dataset_root)

        # Print dataset statistics
        print(f"Number of images for RealWorld360SDInference: {len(self.image_list)}")

    def _generate_image_list(self, cfg, dataset_root):
        """
        Generates a list of image file paths for top and bottom images.

        Parameters:
        - cfg: Configuration object containing dataset settings.
        - dataset_root: Root directory of the dataset.

        Returns:
        - List[List[str]]: List of [bottom, top] image file paths.
        """
        image_top_list = [
            os.path.join(dataset_root, 'image_up', f"{name}.png") for name in cfg.images
        ]
        image_bottom_list = [
            img.replace('up', 'down') for img in image_top_list
        ]

        # Combine bottom and top image paths into a single list
        return [[bottom, top] for bottom, top in zip(image_bottom_list, image_top_list)]
