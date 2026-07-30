import os
import sys
import hashlib
import cv2
import numpy as np

# ==================================================================
# 🛠️ SYSTEM RUNTIME PATH INTEGRATION (same as your other scripts)
# ==================================================================
SCRIPT_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_CURRENT_DIR)

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just saving to file
import matplotlib.pyplot as plt

CLASS_NAMES = ["Healthy", "Blight", "Phyllosticta"]
BASE_DIR = os.path.dirname(SCRIPT_CURRENT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "EfficientNetV2_Cardamom_FineTuned.h5")

DATASET_ROOT = os.path.join(BASE_DIR, "paper")

# Maps: actual folder name on disk -> label name used by the model's CLASS_NAMES
MAPPING = {
    "Healthy": os.path.join(DATASET_ROOT, "Healthy"),
    "Blight": os.path.join(DATASET_ROOT, "LeafBlight"),
    "Phyllosticta": os.path.join(DATASET_ROOT, "LeafSpot")
}

# ==================================================================
# ✅ STEP 1: VERIFY THE .h5 FILE IS BEING USED (not random init)
# ==================================================================
print("==================================================================")
print("🔍 MODEL WEIGHT VERIFICATION")
print("==================================================================")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at: {MODEL_PATH}")
    sys.exit()

# File-level proof: hash + size, so you know exactly which .h5 was loaded
file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
md5_hash = hashlib.md5()
with open(MODEL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        md5_hash.update(chunk)
print(f"📄 Model file          : {MODEL_PATH}")
print(f"📦 File size            : {file_size_mb:.2f} MB")
print(f"🔑 MD5 checksum         : {md5_hash.hexdigest()}")

model = tf.keras.models.load_model(MODEL_PATH)

print("------------------------------------------------------------------")
print("🏗️  MODEL ARCHITECTURE (last 5 layers):")
for layer in model.layers[-5:]:
    print(f"   {layer.name}  ({layer.__class__.__name__})  ->  output shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'n/a'}")
print("------------------------------------------------------------------")

# Weight-level proof: a freshly initialized (untrained) layer has near-zero-mean,
# small-variance, symmetric random weights. A trained layer's weights should show
# non-trivial structure (larger spread, non-symmetric stats, non-zero biases in
# later layers). We sample the final Dense (classifier) layer since that's the
# clearest signal of "did this model actually learn something."
final_layer = model.layers[-1]
w, b = final_layer.get_weights()
print("------------------------------------------------------------------")
print(f"🧠 Final layer name      : {final_layer.name}")
print(f"🧠 Final layer weights   : shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}")
print(f"🧠 Final layer biases    : {b}")
print("   (If these were untrained/random-init, std would typically be a")
print("    tiny fixed default and biases would be exactly 0.0 for all classes.")
print("    Non-zero, non-uniform biases and non-trivial std strongly indicate")
print("    trained weights were loaded successfully.)")
print("==================================================================\n")

feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# ==================================================================
# 📥 STEP 2: LOAD FULL DATASET (all ~1500+ images, ground truth known)
# ==================================================================
samples = []
for true_label, folder_path in MAPPING.items():
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: folder missing, skipping -> {folder_path}")
        continue
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            samples.append({"path": os.path.join(folder_path, f), "true_label": true_label})

if not samples:
    print("❌ No images found. Check dataset paths.")
    sys.exit()

print(f"📊 Total images to evaluate: {len(samples)}\n")

# ==================================================================
# 🔁 STEP 3: RUN INFERENCE ACROSS THE ENTIRE DATASET
# ==================================================================
y_true = []
y_pred = []
failed = []

for i, sample in enumerate(samples):
    img_raw = cv2.imread(sample["path"])
    if img_raw is None:
        failed.append(sample["path"])
        continue

    img_resized = cv2.resize(img_raw, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # cv2 loads as BGR, model expects RGB
    x = img_rgb.astype('float32')
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x_batch = np.expand_dims(x, axis=0)

    raw_output = model.predict(x_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(raw_output))

    y_true.append(sample["true_label"])
    y_pred.append(CLASS_NAMES[predicted_idx])

    if (i + 1) % 100 == 0 or (i + 1) == len(samples):
        print(f"   ...processed {i + 1}/{len(samples)}")

if failed:
    print(f"\n⚠️  {len(failed)} images failed to load and were skipped.")

# ==================================================================
# 📈 STEP 4: ACCURACY, CLASSIFICATION REPORT, CONFUSION MATRIX
# ==================================================================
overall_acc = accuracy_score(y_true, y_pred)

print("\n==================================================================")
print(f"✅ OVERALL ACCURACY: {overall_acc * 100:.2f}%  ({len(y_true)} images evaluated)")
print("==================================================================")

print("\n📋 CLASSIFICATION REPORT (per-class precision/recall/F1):")
print(classification_report(y_true, y_pred, labels=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
print("🧮 CONFUSION MATRIX (rows = true label, cols = predicted label):")
print("           " + "  ".join(f"{c:>13}" for c in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:>10} " + "  ".join(f"{v:>13}" for v in row))

# ==================================================================
# 🖼️ STEP 5: SAVE CONFUSION MATRIX AS AN IMAGE
# ==================================================================
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Accuracy: {overall_acc*100:.2f}%)")

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

fig.colorbar(im)
plt.tight_layout()
output_path = os.path.join(SCRIPT_CURRENT_DIR, "confusion_matrix.png")
plt.savefig(output_path, dpi=150)
print(f"\n🖼️  Confusion matrix image saved to: {output_path}")
print("==================================================================\n")