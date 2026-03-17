import base64
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from model import build_model
from segmentation import segment_vessels
from ransac_refinement import refine_vessels_ransac
from preprocessing import get_val_transforms, normalize_4_channel, preprocess_image_rgb

app = FastAPI(title="DR Classification API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DIAGNOSIS_MAP = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_weights_path = 'checkpoints/best_model.pth'
in_chans_model = 3 # From main.py, fast-trained demo model expects 3 channels

# Load Model Globally
model = None

@app.on_event("startup")
def load_model():
    global model
    model = build_model(num_classes=5, in_chans=in_chans_model, device=device)
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_weights_path}")
    else:
        print("WARNING: Weights file not found. Running with un-finetuned / randomly initialized weights.")
    model.eval()

def encode_image_base64(img_arr: np.ndarray) -> str:
    """Encodes a numpy array image (BGR or Grayscale) to base64 JPEG."""
    retval, buffer = cv2.imencode('.jpg', img_arr)
    if retval:
        return base64.b64encode(buffer).decode('utf-8')
    return ""

@app.post("/api/predict")
async def predict_dr(file: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        return {"error": "Invalid image file."}
        
    # Apply Resize limit (From main.py max_dim = 800)
    h, w = image_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        
    # Segment Vessels
    print("Extracting vessel map using Frangi filter...")
    candidate_mask = segment_vessels(image_bgr)
    print("Refining vessels using RANSAC...")
    refined_mask = refine_vessels_ransac(candidate_mask)
    
    # Classification preprocessing
    # main.py does: pil_rgb = preprocess_image_rgb(image_path, img_size=224) 
    # But we don't have an image path here, so we must adapt preprocess_image_rgb.
    # preprocess_image_rgb internally reads PIL image. We can write a temporary file or use PIL directly.
    import tempfile
    from PIL import Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        # Save BGR to RGB temporarily for preprocess_image_rgb if needed? Actually preprocess_image_rgb takes path.
        cv2.imwrite(tmp.name, image_bgr)
        tmp_path = tmp.name
        
    try:
        pil_rgb = preprocess_image_rgb(tmp_path, img_size=224)
        rgb_tensor = get_val_transforms(img_size=224)(pil_rgb)
        
        if in_chans_model == 4:
            mask_resized = cv2.resize(refined_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized / 255.0).unsqueeze(0).float()
            input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
            input_tensor = normalize_4_channel(input_tensor)
        else:
            input_tensor = rgb_tensor
            
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            confidence = float(np.max(probabilities))
            class_idx = int(np.argmax(probabilities))
            
        diagnosis_label = DIAGNOSIS_MAP[class_idx]
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    # Base64 encodings for the dashboard
    original_b64 = encode_image_base64(image_bgr)
    vessel_b64 = encode_image_base64(candidate_mask)
    refined_b64 = encode_image_base64(refined_mask)
    
    return {
        "prediction": diagnosis_label,
        "confidence": confidence,
        "class_idx": class_idx,
        "original_image": f"data:image/jpeg;base64,{original_b64}",
        "vessel_mask": f"data:image/jpeg;base64,{vessel_b64}",
        "refined_mask": f"data:image/jpeg;base64,{refined_b64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
