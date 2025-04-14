
# [CVPR 2025] <span style="font-variant: small-caps;">Helvipad</span>: A Real-World Dataset for Omnidirectional Stereo Depth Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2411.18335-b31b1b.svg)](https://arxiv.org/abs/2411.18335)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://huggingface.co/datasets/chcorbi/helvipad)
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

## News

- **[14 Apr 2025]** We have released the code of our 360-IGEV-Stereo model which adapts a standard stereo matching architecture to omnidirectional imagery.
- **[08 Apr 2025]** Our new paper **DFI-OmniStereo** achieves state-of-the-art results on Helvipad. Check out the [project page](https://vita-epfl.github.io/DFI-OmniStereo-website/) for details, paper and code.
- **[16 Mar 2025 - CVPR Update]** A small but important update has been applied to the dataset. If you have already downloaded it, please check the details on the [HuggingFace Hub](https://github.com/vita-epfl/helvipad/releases).
- **[16 Feb 2025]** Helvipad has been accepted to CVPR 2025! 🎉🎉

## Dataset Structure

The dataset is organized into training, validation and testing subsets with the following structure:

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
├── val/
│   ├── depth_maps                # Depth maps generated from LiDAR data
│   ├── depth_maps_augmented      # Augmented depth maps using depth completion
│   ├── disparity_maps            # Disparity maps computed from depth maps
│   ├── disparity_maps_augmented  # Augmented disparity maps using depth completion
│   ├── images_top                # Top-camera RGB images
│   ├── images_bottom             # Bottom-camera RGB images
│   ├── LiDAR_pcd                 # Original LiDAR point cloud data
├── test/
│   ├── depth_maps                # Depth maps generated from LiDAR data
│   ├── depth_maps_augmented      # Augmented depth maps using depth completion (only for computing LRCE)
│   ├── disparity_maps            # Disparity maps computed from depth maps
│   ├── disparity_maps_augmented  # Augmented disparity maps using depth completion (only for computing LRCE)
│   ├── images_top                # Top-camera RGB images
│   ├── images_bottom             # Bottom-camera RGB images
│   ├── LiDAR_pcd                 # Original LiDAR point cloud data
```


## Benchmark

We evaluate the performance of multiple state-of-the-art and popular stereo matching methods, both for standard and 360° images. All models are trained on a single NVIDIA A100 GPU with
the largest possible batch size to ensure comparable use of computational resources.

| Method                                              | Stereo Setting    | Disp-MAE (°) | Disp-RMSE (°) | Disp-MARE | Depth-MAE (m) | Depth-RMSE (m) | Depth-MARE | Depth-LRCE (m) |
|-----------------------------------------------------|-------------------|--------------|-----------|-----------|---------------|----------------|------------|----------------|
| [PSMNet](https://arxiv.org/abs/1803.08669)          | conventional      | 0.286        | 0.496     | 0.248     | 2.509         | 5.673          | 0.176      | 1.809          |
| [360SD-Net](https://arxiv.org/abs/1911.04460)       | omnidirectional   | 0.224        | 0.419     | 0.191     | 2.122         | 5.077          | 0.152      | 0.904          |
| [IGEV-Stereo](https://arxiv.org/abs/2303.06615)     | conventional      | 0.225        | 0.423     | 0.172     | 1.860         | 4.447          | 0.146      | 1.203          |
| [360-IGEV-Stereo](https://arxiv.org/abs/2411.18335) | omnidirectional   | 0.188        | 0.404     | 0.146     | 1.720         | 4.297          | 0.130      | **0.388**      |
| [DFI-OmniStereo](https://arxiv.org/abs/2503.23502)  | omnidirectional   | **0.158**    | **0.338** | **0.120** | **1.463**     | **3.767**      | **0.108**  | 0.397          |


## Download

The dataset is available on [HuggingFace Hub](https://github.com/vita-epfl/helvipad/releases).

## 360-IGEV-Stereo

The code of the method 360-IGEV-Stereo can be found in the `360_igev_stereo` directory. We use [Hydra](https://hydra.cc/) for configuration management and [Weights & Biases](https://wandb.ai/site/) for comprehensive experiment tracking and visualization.

### Installation

We assume that the Helvipad dataset has been downloaded and is stored at the location `./data/helvipad`.

#### 1. Set up the environment

```bash
conda create -n 360-igev-stereo python=3.11
conda activate 360-igev-stereo
git clone git@github.com:vita-epfl/Helvipad.git
cd Helvipad/360_igev_stereo
pip install -r requirements.txt
```

#### 2. Download the pretrained weights

##### A. For training
IGEV-Stereo (SceneFlow weights): Create the directory and download the pretrained SceneFlow weights from the [IGEV-Stereo Google Drive](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX), as provided by [IGEV-Stereo](https://github.com/gangweix/IGEV):
```bash
mkdir -p ./models/_360_igev_stereo/pretrained_models/igev_stereo
```
Place the downloaded file into the directory created above.

##### B. For evaluation and inference
360-IGEV-Stereo main checkpoint: Download our pretrained model checkpoint:
```bash
mkdir -p ./models/_360_igev_stereo/pretrained_models/360_igev_stereo && \
wget -O ./models/_360_igev_stereo/pretrained_models/360_igev_stereo/360_igev_stereo_helvipad.pth "https://github.com/vita-epfl/Helvipad/releases/download/v0.1.0/360_igev_stereo_helvipad.pth"
```

### Training

To train the model from the IGEV-Stereo weights, run the following command:
```bash
cd 360_igev_stereo
python train.py \
  --debug=false \
  --exp_name=Main \
  --dataset_root=./data/helvipad/
```
All other parameters are set to their default values for training the main model.

### Evaluation

To evaluate our model using the main checkpoint and compute all metrics including Left-Right Consistency Error (LRCE), use:
```bash
cd src
python evaluate.py \
  --debug=false \
  --exp_name=Evaluation \
  --dataset_root=./data/helvipad/ \
  --restore_ckpt=./models/_360_igev_stereo/pretrained_models/360_igev_stereo/360_igev_stereo_helvipad.pth \
  --calc_lrce=true
```
Note: Setting `--calc_lrce=true` enables LRCE evaluation, which increases computation time.

### Inference

#### Helvipad examples

To generate inference results on selected samples from the Helvipad dataset, run the following command:
```bash
cd src
python infer.py \
  --infer_name=helvipad_results \
  --dataset_root=./data/helvipad/ \
  --restore_ckpt=./models/_360_igev_stereo/pretrained_models/360_igev_stereo/360_igev_stereo_helvipad.pth \
  --images test-20240120_REC_06_IN-0042 test-20240124_REC_03_OUT-0676 test-20240124_REC_08_NOUT-0717
```
This command will process the following frames (all of which are part of the `test` set):
- `0042` from the scene `20240120_REC_06_IN`
- `0676` from the scene `20240124_REC_03_OUT`
- `0717` from the scene `20240124_REC_08_NOUT`

The results as well as the top and bottom images will be saved to: `./models/_360_igev_stereo/inference_results/helvipad_results`.

#### 360SD-Net real-world examples

To evaluate our model on real-world examples from the [360SD-Net](https://github.com/albert100121/360SD-Net) dataset:
1. Download the real-world top and bottom images from the [official repo](https://github.com/albert100121/360SD-Net/tree/master/data/realworld).
2. Place the data in a directory of your choice, e.g., `./data/360sd`.
3. Run the following command to perform inference:
```bash
cd src
python infer.py \
  --infer_name=360SD_results \
  --dataset_root=./data/360sd/ \
  --restore_ckpt=./models/_360_igev_stereo/pretrained_models/360_igev_stereo/360_igev_stereo_helvipad.pth \
  --dataset=360SD \
  --min_disp_deg=0.0048 \
  --max_disp_deg=178 \
  --max_disp=512 \
  --images hall room stairs
```
This will run inference on the following scenes:
- `hall`
- `room`
- `stairs`

The results will be saved in: `./models/_360_igev_stereo/inference_results/360SD_results`.

## Project Page

For more information, visualizations, and updates, visit the **[project page](https://vita-epfl.github.io/Helvipad/)**.

## License

This dataset is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgments

This work was supported by the [EPFL Center for Imaging](https://imaging.epfl.ch/) through a Collaborative Imaging Grant. 
We thank the VITA lab members for their valuable feedback, which helped to enhance the quality of this manuscript. 
We also express our gratitude to Dr. Simone Schaub-Meyer and Oliver Hahn for their insightful advice during the project's final stages.

## Citation

If you use the Helvipad dataset in your research, please cite our paper:

```bibtex
@inproceedings{zayene2025helvipad,
  author    = {Zayene, Mehdi and Endres, Jannik and Havolli, Albias and Corbière, Charles and Cherkaoui, Salim and Ben Ahmed Kontouli, Alexandre and Alahi, Alexandre},
  title     = {Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
