import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# Paths
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "EfficientNetV2_Cardamom_FineTuned.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Healthy", "Blight", "Phyllosticta"]

DATASET_ROOT = os.path.join(BASE_DIR, "Cardamom_Plant_Dataset_Chinnahalli_1724")
BG_DIR = os.path.join(DATASET_ROOT, "BG_REMOVED")

MAPPING = {
    "Healthy": os.path.join(BG_DIR, "Background_Removed_Healthy"),
    "Blight": os.path.join(BG_DIR, "Background_Removed_Blight1000"),
    "Phyllosticta": os.path.join(BG_DIR, "Background_Removed_Phyllosticta_Leaf_Spot")
}

# ==========================================================
# Statistics
# ==========================================================
correct = 0
total = 0

class_correct = {c: 0 for c in CLASS_NAMES}
class_total = {c: 0 for c in CLASS_NAMES}

all_true = []
all_pred = []

print(f"{'Image':<45} {'True':>15} {'Predicted':>15} {'Conf':>10} {'Result':>10}")
print("-" * 105)

# ==========================================================
# Evaluate every image
# ==========================================================
for true_class, folder in MAPPING.items():

    if not os.path.exists(folder):
        print(f"Missing folder: {folder}")
        continue

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"\nProcessing {len(files)} images from {true_class}...\n")

    for fname in files:

        path = os.path.join(folder, fname)

        try:
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img).astype("float32") / 255.0
            x = np.expand_dims(x, axis=0)

            probs = model.predict(x, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            is_correct = pred_class == true_class

            total += 1
            if is_correct:
                correct += 1
                class_correct[true_class] += 1

            class_total[true_class] += 1

            all_true.append(true_class)
            all_pred.append(pred_class)

            status = "OK" if is_correct else "WRONG"

            print(
                f"{fname:<45} "
                f"{true_class:>15} "
                f"{pred_class:>15} "
                f"{confidence:>8.2f}% "
                f"{status:>10}"
            )

        except Exception as e:
            print(f"Error processing {fname}: {e}")

# ==========================================================
# Overall Results
# ==========================================================
print("\n" + "=" * 105)

if total > 0:
    overall_accuracy = (correct / total) * 100
    print(f"Overall Accuracy : {correct}/{total} = {overall_accuracy:.2f}%")
else:
    print("No images were evaluated.")

print("\nPer-Class Accuracy")

for cls in CLASS_NAMES:
    if class_total[cls] > 0:
        acc = (class_correct[cls] / class_total[cls]) * 100
        print(f"{cls:<15}: {class_correct[cls]}/{class_total[cls]} = {acc:.2f}%")

# ==========================================================
# Confusion Matrix
# ==========================================================
print("\nConfusion Matrix (Rows=True, Columns=Predicted)\n")

cm = confusion_matrix(all_true, all_pred, labels=CLASS_NAMES)

print(f"{'':>18}{'Healthy':>12}{'Blight':>12}{'Phyllosticta':>16}")

for i, cls in enumerate(CLASS_NAMES):
    print(
        f"{cls:>18}"
        f"{cm[i][0]:>12}"
        f"{cm[i][1]:>12}"
        f"{cm[i][2]:>16}"
    )

# ==========================================================
# Classification Report
# ==========================================================
print("\nClassification Report\n")

print(
    classification_report(
        all_true,
        all_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        digits=4
    )
)