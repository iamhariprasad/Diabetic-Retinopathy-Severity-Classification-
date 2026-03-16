import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def extract_green_channel(image_bgr):
    """
    Extract the green channel from a BGR image.
    Retinal vessels are most prominent in the green channel.
    """
    b, g, r = cv2.split(image_bgr)
    return g

def apply_clahe(image_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances the contrast of the retinal vessels.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_img = clahe.apply(image_gray)
    return enhanced_img

def preprocess_for_segmentation(image_bgr):
    """
    Preprocess an image for classical vessel segmentation.
    Includes Green channel extraction and CLAHE.
    """
    green_channel = extract_green_channel(image_bgr)
    enhanced_green = apply_clahe(green_channel)
    return enhanced_green

def get_train_transforms(img_size=224):
    """
    Data augmentation for training images.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # Normalize with ImageNet stats for RGB channels, 0.5 for the 4th (mask) channel 
        # (This will be applied later for 4-channel tensor, or we just normalize RGB 
        # and leave mask scaled between 0-1)
    ])

def get_val_transforms(img_size=224):
    """
    Transforms for validation/testing images.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def preprocess_image_rgb(image_path, img_size=224):
    """
    Loads an image, coverts to RGB, and resizes it.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # PIL image for torchvision transforms
    pil_img = Image.fromarray(img_rgb)
    return pil_img

def normalize_4_channel(tensor_4c):
    """
    Normalize a 4-channel tensor (RGB + Mask).
    RGB uses ImageNet mean/std, Mask uses mean=0.5, std=0.5 roughly.
    """
    # ImageNet mean and std for RGB
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(-1, 1, 1)
    
    return (tensor_4c - mean) / std
