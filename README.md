
# <span style="font-variant: small-caps;">Helvipad</span>: A Real-World Dataset for Omnidirectional Stereo Depth Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2411.18335-b31b1b.svg)](https://arxiv.org/abs/2411.18335)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://github.com/vita-epfl/helvipad/releases)
[![Project Page](https://img.shields.io/badge/Project-Page-brightgreen)](https://vita-epfl.github.io/Helvipad/)

![Front Page](static/images/front_page.png)
## Abstract

Despite considerable progress in stereo depth estimation, omnidirectional imaging remains underexplored,
mainly due to the lack of appropriate data.
We introduce <span style="font-variant: small-caps;">Helvipad</span>,
a real-world dataset for omnidirectional stereo depth estimation, consisting of 40K frames from video sequences
across diverse environments, including crowded indoor and outdoor scenes with diverse lighting conditions.
Collected using two 360° cameras in a top-bottom setup and a LiDAR sensor, the dataset includes accurate
depth and disparity labels by projecting 3D point clouds onto equirectangular images. Additionally, we
provide an augmented training set with a significantly increased label density by using depth completion.
We benchmark leading stereo depth estimation models for both standard and omnidirectional images.
The results show that while recent stereo methods perform decently, a significant challenge persists in accurately
estimating depth in omnidirectional imaging. To address this, we introduce necessary adaptations to stereo models,
achieving improved performance.

## Dataset Structure

The dataset is organized into training and testing subsets with the following structure:

```
helvipad/
├── train/
│   ├── depth_maps                # Depth maps generated from LiDAR data
│   ├── depth_maps_augmented      # Augmented depth maps using depth completion
│   ├── disparity_maps            # Disparity maps computed from depth maps
│   ├── disparity_maps_augmented  # Augmented disparity maps using depth completion
│   ├── images_top                # Top-camera RGB images
│   ├── images_bottom             # Bottom-camera RGB images
│   ├── LiDAR_pcd                 # Original LiDAR point cloud data
├── test/
│   ├── depth_maps                # Depth maps generated from LiDAR data
│   ├── disparity_maps            # Disparity maps computed from depth maps
│   ├── images_top                # Top-camera RGB images
│   ├── images_bottom             # Bottom-camera RGB images
│   ├── LiDAR_pcd                 # Original LiDAR point cloud data
```


## Benchmark

We evaluate the performance of multiple state-of-the-art and popular stereo matching methods, both for standard and 360° images. All models are trained on a single NVIDIA A100 GPU with
the largest possible batch size to ensure comparable use of computational resources.

| Method             | Type           | Disp-MAE (°) | Disp-RMSE (°) | Disp-MARE | Depth-MAE (m) | Depth-RMSE (m) | Depth-MARE (m) |
|--------------------|----------------|--------------|---------------|-----------|---------------|----------------|----------------|
| [PSMNet](https://arxiv.org/abs/1803.08669)           | Stereo        | 0.33         | 0.54          | 0.20      | 2.79          | 6.17           | 0.29           |
| [360SD-Net](https://arxiv.org/abs/1911.04460)        | 360° Stereo   | 0.21         | 0.42          | 0.18      | 2.14          | 5.12           | 0.15           |
| [IGEV-Stereo](https://arxiv.org/abs/2303.06615)      | Stereo        | 0.22         | 0.41          | 0.17      | 1.85          | 4.44           | 0.15           |
| 360-IGEV-Stereo    | 360° Stereo   | **0.18**     | **0.39**      | **0.15**  | **1.77**      | **4.36**       | **0.14**       |

## Download

The dataset will be soon available for download [here](https://github.com/vita-epfl/helvipad/releases).


## Project Page

For more information, visualizations, and updates, visit the **[project page](https://vita-epfl.github.io/Helvipad/)**.

## Citation

If you use the Helvipad dataset in your research, please cite our paper:

```bibtex
@misc{zayene2024helvipad,
  author        = {Zayene, Mehdi and Endres, Jannik and Havolli, Albias and Corbière, Charles and Cherkaoui, Salim and Ben Ahmed Kontouli, Alexandre and Alahi, Alexandre},
  title         = {Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation},
  year          = {2024},
  eprint        = {2403.16999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

This dataset is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgments

This work was supported by the [EPFL Center for Imaging](https://imaging.epfl.ch/) through a Collaborative Imaging Grant. 
We thank the VITA lab members for their valuable feedback, which helped to enhance the quality of this manuscript. 
We also express our gratitude to Dr. Simone Schaub-Meyer and Oliver Hahn for their insightful advice during the project's final stages.