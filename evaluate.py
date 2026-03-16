import torch
import numpy as np
from tqdm import tqdm
from utils import calculate_classification_metrics, calculate_segmentation_metrics, plot_confusion_matrix
import os

def evaluate_model(model, dataloader, criterion, device, save_cm=False, save_dir='results'):
    """
    Evaluates the classification model on a given dataset (validation/test).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    acc, prec, rec, f1 = calculate_classification_metrics(all_labels, all_preds)
    
    if save_cm:
        os.makedirs(save_dir, exist_ok=True)
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_preds, classes=[0,1,2,3,4], save_path=cm_path)
        
    return avg_loss, acc, f1

def evaluate_segmentation_pipeline(test_img_dir, test_mask_dir, segmenter_fn):
    """
    Evaluates the classical vessel segmentation + RANSAC refinement pipeline.
    Uses the DRIVEDataset or custom loop to compute Dice and IoU on test set.
    """
    import cv2
    from PIL import Image
    
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.tif') or f.endswith('.png')]
    all_dice = []
    all_iou = []
    
    print(f"Evaluating Pipeline on {len(test_images)} Segmentation Samples...")
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_img_dir, img_name)
        
        # Match mask nomenclature for DRIVE
        base_name = img_name.split('_')[0]
        mask_path = os.path.join(test_mask_dir, f"{base_name}_manual1.gif")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(test_mask_dir, img_name)
             
        img_bgr = cv2.imread(img_path)
        mask_pil = Image.open(mask_path).convert('L')
        mask_gt = np.array(mask_pil) / 255.0  # target: 0 to 1
        mask_gt = (mask_gt > 0.5).astype(np.uint8)
        
        # Generate prediction
        pred_mask_255 = segmenter_fn(img_bgr)
        pred_mask = (pred_mask_255 > 127).astype(np.uint8)
        
        # Calculate metrics
        dice, iou = calculate_segmentation_metrics(mask_gt, pred_mask)
        all_dice.append(dice)
        all_iou.append(iou)
        
    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    
    print(f"Segmentation Evaluation Complete:")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    
    return avg_dice, avg_iou
