import cv2
import numpy as np

def extract_leaf_severity(image_path, leaf_mask):
    """
    Computes the Plant Disease Index (PDI) for a leaf image given its binary mask.
    
    :param image_path: Path to the raw leaf image file.
    :param leaf_mask: A binary numpy array (255 for leaf, 0 for background) 
                      representing the isolated target leaf footprint.
    :return: float (calculated PDI percentage)
    """
    # 1. Load image and convert color space
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Hardcoded Calibrated Dual-Engine Blended Parameters (9.83% MAE Sweet Spot)
    mask_white_pins = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 60, 255]))
    mask_brown_strips = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([30, 255, 219]))
    blended_lesions = cv2.bitwise_or(mask_white_pins, mask_brown_strips)
    
    # 3. Intersect with the provided leaf mask (from U2-Net or Manual)
    final_lesion_mask = cv2.bitwise_and(blended_lesions, leaf_mask)
    
    # 4. Biological Noise Filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Pixel Calculus
    total_leaf_pixels = np.count_nonzero(leaf_mask)
    total_lesion_pixels = np.count_nonzero(final_lesion_mask)
    
    if total_leaf_pixels == 0:
        return 0.0
        
    pdi = (total_lesion_pixels / total_leaf_pixels) * 100
    return round(pdi, 4)