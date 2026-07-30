import os
import sys
import cv2
import numpy as np

# ==================================================================
# PROJECT PATHS
# ==================================================================
SCRIPT_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_CURRENT_DIR)

# Add WORKSPACE_ROOT to sys.path so we can import modules sitting in the root directory
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# ==================================================================
# IMPORT SEVERITY ENGINES
# ==================================================================
from phyllosticta_engine import extract_leaf_severity
from blight_engine import extract_blight_severity

# ==================================================================
# FRAMEWORK IMPORTS
# ==================================================================
import tensorflow as tf

CLASS_NAMES = ["Healthy", "Blight", "Phyllosticta"]

BASE_DIR = os.path.dirname(SCRIPT_CURRENT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "EfficientNetV2_Cardamom_FineTuned.h5")

# ==================================================================
# COMMAND LINE INPUT
# ==================================================================

if len(sys.argv) != 2:
    print("\nUsage:")
    print("python test_blind_inference.py <image_path>")
    print("\nExample:")
    print("python test_blind_inference.py test.jpg")
    sys.exit(1)

selected_file_path = sys.argv[1]

if not os.path.exists(selected_file_path):
    print(f"\nError: File not found:\n{selected_file_path}")
    sys.exit(1)

filename = os.path.basename(selected_file_path)
parent_folder_name = os.path.basename(os.path.dirname(selected_file_path)).lower()

if "healthy" in parent_folder_name:
    true_label = "Healthy"
elif "blight" in parent_folder_name:
    true_label = "Blight"
elif "phyllosticta" in parent_folder_name or "phylosticta" in parent_folder_name:
    true_label = "Phyllosticta"
else:
    true_label = "Unknown"

# ==================================================================
# LOAD MODEL
# ==================================================================

print("\nLoading EfficientNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

print("Model loaded successfully.")

# ==================================================================
# PREPROCESS IMAGE
# ==================================================================

img_raw = cv2.imread(selected_file_path)

if img_raw is None:
    print("Failed to read image.")
    sys.exit(1)

img_resized = cv2.resize(img_raw, (224, 224))

x = img_resized.astype("float32") / 255.0
x_batch = np.expand_dims(x, axis=0)

# ==================================================================
# CLASSIFICATION
# ==================================================================

features = feature_extractor.predict(x_batch, verbose=0)

weights, biases = model.layers[-1].get_weights()

raw_logits = np.dot(features, weights)[0] + biases

probabilities = tf.nn.softmax(raw_logits).numpy()

predicted_class_idx = int(np.argmax(raw_logits))
predicted_class_name = CLASS_NAMES[predicted_class_idx]

# ==================================================================
# LEAF MASK (Placeholder)
# ==================================================================

gray_leaf = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

_, predicted_leaf_mask = cv2.threshold(
    gray_leaf,
    20,
    255,
    cv2.THRESH_BINARY
)

kernel_clean = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

predicted_leaf_mask = cv2.morphologyEx(
    predicted_leaf_mask,
    cv2.MORPH_CLOSE,
    kernel_clean
)

# ==================================================================
# SEVERITY ANALYSIS
# ==================================================================

calculated_pdi = 0.0

if predicted_class_name == "Phyllosticta":
    calculated_pdi = extract_leaf_severity(
        selected_file_path,
        predicted_leaf_mask
    )

elif predicted_class_name == "Blight":
    calculated_pdi = extract_blight_severity(
        selected_file_path,
        predicted_leaf_mask
    )

else:
    calculated_pdi = 0.0

# ==================================================================
# SEVERITY LABEL
# ==================================================================

if calculated_pdi == 0.0:
    severity_tier = "None (Healthy Tissue Signature)"

elif calculated_pdi < 2.0:
    severity_tier = "Mild Pathology Spread"

elif calculated_pdi < 7.0:
    severity_tier = "Moderate Pathology Spread"

else:
    severity_tier = "Severe Pathology Devastation"

# ==================================================================
# REPORT
# ==================================================================

print("\n==================================================================")

if predicted_class_name.lower() == true_label.lower():
    print("🎉 VERIFICATION SUCCESS: The AI model correctly identified the leaf!")
else:
    print("❌ VERIFICATION MISMATCH: The AI misidentified this sample.")

print("==================================================================")

print(f"📁 Ground Truth Category           : {true_label}")
print(f"📄 Selected Image                  : {filename}")

print("------------------------------------------------------------------")

print("🚨 RAW MODEL LOGITS")

for idx, class_label in enumerate(CLASS_NAMES):
    print(
        f"{class_label:<15}: {raw_logits[idx]:.4f}"
    )

print("------------------------------------------------------------------")

print("📊 SOFTMAX PROBABILITIES")

for idx, class_label in enumerate(CLASS_NAMES):
    print(
        f"{class_label:<15}: {probabilities[idx]*100:.2f}%"
    )

print("==================================================================")

print("🌿 DISEASE ANALYSIS")

print(f"Predicted Disease : {predicted_class_name}")
print(f"PDI               : {calculated_pdi}%")
print(f"Severity          : {severity_tier}")

print("==================================================================\n")