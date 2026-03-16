# Diabetic Retinopathy (DR) Detection Pipeline

An end-to-end pipeline for predicting Diabetic Retinopathy severity from retinal images. This project combines classical computer vision techniques for vessel segmentation with a modern Vision Transformer (ViT) architecture for classification.

## Overview

The pipeline explicitly extracts retinal vessels to assist in the explainability of the model. The detection process includes:
1. **Frangi Filter**: Used to extract a candidate vessel map from the retinal image.
2. **RANSAC Refinement**: Refines the extracted vessels to remove noise and improve the segmentation map.
3. **Vision Transformer (ViT)**: A deep learning classifier that processes the retinal image to output a diagnosis severity.

### Diagnosis Categories
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## Important Note
**We have trained and tested on large data and results are rational, interpretable, and comparable.**

## Project Structure
- `main.py`: The entry point for running end-to-end inference on a single image.
- `model.py`: ViT model architecture definitions.
- `segmentation.py`: Classical vessel segmentation using the Frangi filter.
- `ransac_refinement.py`: RANSAC algorithms for cleaning the segmentation mask.
- `preprocessing.py`: Image transforms and normalization.
- `train.py` & `evaluate.py`: Scripts for training and evaluating the network.
- `utils.py`: Utility functions, including tools for visualization.

## Requirements

Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- NumPy
- Pillow

You can install the required packages using pip:
```bash
pip install torch torchvision torchaudio opencv-python numpy Pillow
```

## Usage

To run inference on a single retinal image:

```bash
python main.py path/to/your/image.png --weights checkpoints/best_model.pth --output_file inference_result.txt
```

### Arguments:
- `image_path`: Path to the input retinal image (supports `.png`, `.jpeg`, `.jpg`, `.tif`).
- `--weights`: Path to the trained model weights (default: `checkpoints/best_model.pth`).
- `--output_file`: Path to save the prediction textual results (default: `inference_result.txt`).

Upon completion, the terminal will output the predicted DR severity and confidence score. It will also display a visualization comprising the original image alongside the unrefined and refined vessel masks.
