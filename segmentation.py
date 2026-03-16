import cv2
import numpy as np
from skimage.filters import frangi

from preprocessing import preprocess_for_segmentation

def segment_vessels(image_bgr):
    """
    Complete classical pipeline for generating a vessel candidate mask.
    1. Preprocess (Green Channel + CLAHE)
    2. Gaussian Smoothing
    3. Frangi Filter (Vessel enhancement)
    4. Thresholding to create binary mask
    """
    # 1. Preprocess
    enhanced_green = preprocess_for_segmentation(image_bgr)
    
    # Create Field of View (FOV) mask to exclude the sharp background boundaries
    # which dominate the Frangi filter response scale
    _, fov_mask = cv2.threshold(enhanced_green, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    fov_mask = cv2.erode(fov_mask, kernel, iterations=1)
    
    # 2. Gaussian Smoothing to remove high frequency noise
    smoothed = cv2.GaussianBlur(enhanced_green, (5, 5), 0)
    
    # Convert to float representation for skimage
    smoothed_float = smoothed.astype(np.float32) / 255.0
    
    # 3. Frangi Filter extracts tube-like structures
    vessel_response = frangi(smoothed_float, sigmas=range(1, 4, 1), black_ridges=True)
    
    # Mask out the edge artifacts outside the core retina
    vessel_response[fov_mask == 0] = 0
    
    # 4. Thresholding
    # Normalize frangi response to 0-255
    vessel_response_norm = cv2.normalize(vessel_response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Otsu's thresholding
    _, binary_mask = cv2.threshold(vessel_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_mask
