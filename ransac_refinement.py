import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

def refine_vessels_ransac(binary_mask, min_component_area=20, residual_threshold=2.0, min_inlier_ratio=0.3):
    """
    Refines the vessel mask by removing non-linear/blooby artifacts.
    Retinal vessels are composed of piece-wise linear or curved segments.
    We extract Connected Components, and for each:
      1. Check if area is too small (remove tiny noise).
      2. If large enough, fit a RANSAC line to its pixels.
      3. If the component represents a vessel, the line should fit a good 
         portion of the points (high inlier ratio). If not, it's a blob/lesion (exudate noise).
    """
    # Ensure binary mask is 0 or 255 of type uint8
    if binary_mask.max() == 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
        
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    refined_mask = np.zeros_like(binary_mask)
    
    for label in range(1, num_labels): # Skip 0 (background)
        area = stats[label, cv2.CC_STAT_AREA]
        
        # 1. Size filtering
        if area < min_component_area:
            continue
            
        # Extract pixel coordinates of this component
        y, x = np.where(labels_im == label)
        points = np.column_stack((x, y)) # (N, 2)
        
        # 2. RANSAC geometry fitting
        # If the component is a line/vessel, RANSAC will find a line that
        # hits many points. If it's a circular blob, a line will only hit a few.
        if len(points) > 5:
            # Features (X) and Targets (Y) for 2D line fitting
            X_data = points[:, 0].reshape(-1, 1)
            y_data = points[:, 1].reshape(-1, 1)
            
            # Since a vessel might be vertical (x = c), standard linear regression (y = mx + b) 
            # might fail. We check variance of x and y to decide which is the dependent variable.
            var_x = np.var(X_data)
            var_y = np.var(y_data)
            
            ransac = RANSACRegressor(residual_threshold=residual_threshold, max_trials=50)
            
            try:
                if var_x > var_y:
                    # Line is more horizontal y = f(x)
                    ransac.fit(X_data, y_data)
                else:
                    # Line is more vertical x = f(y)
                    ransac.fit(y_data, X_data)
                    
                inlier_mask = ransac.inlier_mask_
                inlier_ratio = np.sum(inlier_mask) / len(points)
                
                # 3. Decision
                # If a large percentage of pixels follow a linear trend, it's a vessel piece.
                if inlier_ratio >= min_inlier_ratio:
                    # Keep entire component, or just the inliers?
                    # Generally, vessels curve, so piece-wise it might have lower inliers 
                    # than perfectly straight lines. We just want to reject large circular blobs.
                    # So we'll keep the whole component if it passes.
                    refined_mask[y, x] = 255
            except ValueError:
                # E.g., not enough samples or RANSAC fails to find valid consensus
                pass
        else:
            # Component is small but above min_component_area, keep it
            refined_mask[y, x] = 255
            
    return refined_mask

def get_final_vessel_mask(image_bgr):
    """
    End-to-end interface: raw image -> segmented -> ransac refined.
    """
    from segmentation import segment_vessels
    
    candidate_mask = segment_vessels(image_bgr)
    refined_mask = refine_vessels_ransac(candidate_mask)
    return refined_mask
