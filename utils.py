import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def set_seed(seed=42):
    """
    Control pseudo-random Number Generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculates Accuracy, Precision, Recall, and F1-score.
    """
    acc = accuracy_score(y_true, y_pred)
    # Using macro avg for multi-class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def calculate_segmentation_metrics(y_true, y_pred):
    """
    Computes Dice coefficient and IoU.
    Metrics are based on binary masks (0 or 1).
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (y_true.sum() + y_pred.sum() + 1e-6)
    
    return dice, iou

def plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4], save_path=None):
    """
    Renders and optionally saves a heatmap for the Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: APTOS 2019 DR Severity')
    
    if save_path:
        plt.savefig(save_path)
    # plt.show() # Prevent blocking if running unattended
    plt.close()

def display_inference_results(original, vessel_mask, refined_mask, prediction, confidence, true_label=None):
    """
    Visualizes the raw image side-by-side with intermediate and final pipeline steps.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if original is not None:
        axes[0].imshow(original)
        axes[0].set_title('Original Retinal Image')
        axes[0].axis('off')
        
    if vessel_mask is not None:
        axes[1].imshow(vessel_mask, cmap='gray')
        axes[1].set_title('Classical Mask (Frangi)')
        axes[1].axis('off')
        
    if refined_mask is not None:
        axes[2].imshow(refined_mask, cmap='gray')
        axes[2].set_title(f'Refined Mask (RANSAC)')
        axes[2].axis('off')
        
    title = f"Prediction: {prediction} (Conf: {confidence:.2f})"
    if true_label is not None:
        title += f" | True: {true_label}"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# NOTE: For ViT Attention, since we used a simple model wrapped tightly inside timm, 
# you'd need PyTorch-GradCAM library. In standard python:
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# This requires installing pip install grad-cam.
