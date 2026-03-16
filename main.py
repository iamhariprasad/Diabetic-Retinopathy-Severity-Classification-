import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from model import build_model
from segmentation import segment_vessels
from ransac_refinement import refine_vessels_ransac
from preprocessing import get_val_transforms, normalize_4_channel, preprocess_image_rgb
from utils import display_inference_results

# DR Severity Classes
DIAGNOSIS_MAP = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def load_inference_model(weights_path, in_chans=4, device='cpu'):
    """
    Loads the trained ViT model with configurable channel input capability.
    """
    model = build_model(num_classes=5, in_chans=in_chans, device=device)
    if os.path.exists(weights_path):
        # We handle case where weights might be CPU or CUDA
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print("WARNING: Weights file not found. Running with un-finetuned / randomly initialized weights.")
    
    model.eval()
    return model

def run_inference(image_path, model_weights_path='checkpoints/best_model.pth', output_file=None):
    """
    End-to-End inference pipeline combining classical segmentation and ViT classification.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
        
    image_bgr = cv2.imread(image_path)
    
    # Resize image to max 800px on the longest side to ensure the CV algorithms run fast
    h, w = image_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        
    image_rgb_display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # We enable vessel extraction purely for the visualization
    # even though our fast-trained demo model expects 3 channels.
    use_vessels = True
    in_chans = 3
    
    if use_vessels:
        print("Extracting vessel map using Frangi filter (This may take a minute)...")
        candidate_mask = segment_vessels(image_bgr)
        print("Refining vessels using RANSAC...")
        refined_mask = refine_vessels_ransac(candidate_mask)
    else:
        candidate_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        refined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    print(f"Preparing {in_chans}-channel tensor for ViT...")
    pil_rgb = preprocess_image_rgb(image_path, img_size=224)
    rgb_tensor = get_val_transforms(img_size=224)(pil_rgb) # 3x224x224
    
    if in_chans == 4:
        mask_resized = cv2.resize(refined_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized / 255.0).unsqueeze(0).float()
        input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
        input_tensor = normalize_4_channel(input_tensor)
    else:
        # Standard RGB normalization is typically handled in get_val_transforms
        input_tensor = rgb_tensor
        
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    print("Running classification model...")
    model = load_inference_model(model_weights_path, in_chans=in_chans, device=device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        confidence = np.max(probabilities)
        class_idx = np.argmax(probabilities)
        
    diagnosis_label = DIAGNOSIS_MAP[class_idx]
    
    # Optional True label if requested
    true_label = None
    
    print(f"\n--- Inference Complete ---")
    print(f"Prediction: {diagnosis_label} (Class {class_idx})")
    print(f"Confidence: {confidence*100:.2f}%")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Inference Results for {image_path}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Prediction: {diagnosis_label} (Class {class_idx})\n")
            f.write(f"Confidence: {confidence*100:.2f}%\n")
        print(f"Output saved to {output_file}")
    
    # 6. Visualization
    display_inference_results(
        original=image_rgb_display,
        vessel_mask=candidate_mask,
        refined_mask=refined_mask,
        prediction=diagnosis_label,
        confidence=confidence,
        true_label=true_label
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DR Classification + Vessel Segmentation Inference")
    parser.add_argument('image_path', type=str, help='Path to retinal image (.png, .jpeg, .tif)')
    parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth', help='Path to model weights')
    parser.add_argument('--output_file', type=str, default='inference_result.txt', help='Path to save output results')
    
    args = parser.parse_args()
    
    run_inference(args.image_path, args.weights, args.output_file)
