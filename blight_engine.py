import cv2
import numpy as np

def extract_blight_severity(image_path, leaf_mask):
    """
    Computes the Plant Disease Index (PDI) for Blight given its binary mask footprint.
    
    :param image_path: Path to the raw leaf image file.
    :param leaf_mask: A binary numpy array (255 for leaf, 0 for background) 
                      representing the isolated target leaf footprint.
    :return: float (calculated PDI percentage)
    """
    # 1. Load image and convert domains
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Hardcoded Calibrated Dual-Domain Matrix (9.98% MAE Sweet Spot)
    lower_brown = np.array([3, 6, 35])
    upper_brown = np.array([28, 255, 230])
    mask_hsv_lesion = cv2.inRange(hsv, lower_brown, upper_brown)
    
    mask_dark_lesion = (gray < 103) & (leaf_mask > 0)
    
    blended_raw_lesion = cv2.bitwise_or(mask_hsv_lesion, mask_dark_lesion.astype(np.uint8) * 255)
    
    # Intersect findings strictly inside the provided leaf footprint boundary
    final_lesion_mask = cv2.bitwise_and(blended_raw_lesion, leaf_mask)
    
    # 3. Specular Glare / Water Droplet Filter Gate
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    glare_mask = (v_channel > 235) & (s_channel < 35)
    final_lesion_mask[glare_mask] = 0
    
    # 4. Cleanup pass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Pixel Calculus
    total_leaf_pixels = np.count_nonzero(leaf_mask)
    total_lesion_pixels = np.count_nonzero(final_lesion_mask)
    
    if total_leaf_pixels == 0:
        return 0.0
        
    pdi = (total_lesion_pixels / total_leaf_pixels) * 100
    return round(pdi, 4)