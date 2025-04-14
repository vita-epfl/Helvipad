import numpy as np
import random
from PIL import Image
import cv2
from torchvision.transforms import ColorJitter, functional, Compose

# Disable OpenCV threading for better performance in multi-threaded environments
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AdjustGamma:
    """
    Custom transformation to adjust the gamma and gain of an image.
    """

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        """
        Initializes the AdjustGamma transformation.

        Parameters:
        - gamma_min, gamma_max: Range for random gamma adjustment.
        - gain_min, gain_max: Range for random gain adjustment.
        """
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gain_min = gain_min
        self.gain_max = gain_max

    def __call__(self, sample):
        """
        Applies the gamma and gain adjustment to the input image.

        Parameters:
        - sample: Input image (PIL Image or Tensor).

        Returns:
        - Transformed image with adjusted gamma and gain.
        """
        # Randomly sample gain and gamma values within the specified ranges
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        # Apply the gamma adjustment using torchvision's functional API
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        """
        Returns a string representation of the transformation for debugging.
        """
        return f"Adjust Gamma ({self.gamma_min}, {self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class SparseFlowAugmentor:
    """
    Augmentor class for applying photometric transformations to stereo image pairs.
    """

    def __init__(self, do_photo, saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):
        """
        Initializes the SparseFlowAugmentor.

        Parameters:
        - do_photo: Boolean indicating whether photometric augmentation is enabled.
        - saturation_range: Range for random saturation adjustment.
        - gamma: List of gamma adjustment parameters [gamma_min, gamma_max, gain_min, gain_max].
        """
        # Initialize photometric augmentation if enabled
        if do_photo:
            self.photo_aug = Compose([
                ColorJitter(
                    brightness=0.3,  # Random brightness adjustment
                    contrast=0.3,    # Random contrast adjustment
                    saturation=list(saturation_range),  # Random saturation adjustment
                    hue=0.3 / 3.14   # Random hue adjustment
                ),
                AdjustGamma(*gamma)  # Custom gamma adjustment
            ])
        else:
            # No photometric augmentation
            self.photo_aug = None

    def color_transform(self, img1, img2):
        """
        Applies photometric transformations to a pair of stereo images.

        Parameters:
        - img1, img2: Stereo image pair (numpy arrays).

        Returns:
        - Transformed stereo image pair (numpy arrays).
        """
        # Stack the two images along the first axis for joint transformation
        image_stack = np.concatenate([img1, img2], axis=0)

        # Apply photometric augmentation if enabled
        if self.photo_aug is not None:
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)

        # Split the transformed stack back into two images
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def __call__(self, img1, img2, depth, disp, valid):
        """
        Applies the augmentation pipeline to the input data.

        Parameters:
        - img1, img2: Stereo image pair (numpy arrays).
        - depth: Depth map (numpy array).
        - disp: Disparity map (numpy array).
        - valid: Validity mask (numpy array).

        Returns:
        - Augmented img1, img2, depth, disp, and valid.
        """
        # Apply photometric transformations if enabled
        if self.photo_aug is not None:
            img1, img2 = self.color_transform(img1, img2)

        # Return the (possibly augmented) data
        return img1, img2, depth, disp, valid