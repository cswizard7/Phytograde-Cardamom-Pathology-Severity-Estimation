import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random

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

random.seed(42)
correct = 0
total = 0
class_correct = {c: 0 for c in CLASS_NAMES}
class_total = {c: 0 for c in CLASS_NAMES}
confidence_sum = 0

print(f"{'Image':<40} {'True':>12} {'Predicted':>12} {'Confidence':>12} {'Result':>8}")
print("-" * 90)

for true_class, folder in MAPPING.items():
    if not os.path.exists(folder):
        print(f"MISSING FOLDER: {folder}")
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    sample = random.sample(files, min(10, len(files)))
    for fname in sample:
        path = os.path.join(folder, fname)
        try:
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img).astype('float32') / 255.0
            x_batch = np.expand_dims(x, axis=0)
            probs = model.predict(x_batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100
            is_correct = pred_class == true_class
            correct += is_correct
            total += 1
            class_correct[true_class] += is_correct
            class_total[true_class] += 1
            confidence_sum += confidence
            result = "OK" if is_correct else "WRONG"
            print(f"{fname:<40} {true_class:>12} {pred_class:>12} {confidence:>11.2f}% {result:>8}")
        except Exception as e:
            print(f"Error on {fname}: {e}")

print("-" * 90)
print(f"\nOverall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print(f"Mean confidence: {confidence_sum/total:.1f}%")
print(f"\nPer-class accuracy:")
for c in CLASS_NAMES:
    if class_total[c] > 0:
        print(f"  {c}: {class_correct[c]}/{class_total[c]} = {class_correct[c]/class_total[c]*100:.1f}%")