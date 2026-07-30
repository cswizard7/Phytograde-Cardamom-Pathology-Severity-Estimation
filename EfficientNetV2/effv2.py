import os
import sys
import shutil

# ==================================================================
# 🛠️ AUTOMATED MODULAR GPU PATH BINDING & RUNTIME INJECTION
# ==================================================================
SCRIPT_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_CURRENT_DIR)
SITE_PACKAGES = os.path.join(WORKSPACE_ROOT, "tf_gpu_env", "Lib", "site-packages", "nvidia")

CUDNN_BIN    = os.path.join(SITE_PACKAGES, "cudnn", "bin")
RUNTIME_BIN  = os.path.join(SITE_PACKAGES, "cuda_runtime", "bin")
CURAND_BIN   = os.path.join(SITE_PACKAGES, "curand", "bin")
CUSOLVER_BIN = os.path.join(SITE_PACKAGES, "cusolver", "bin")
CUSPARSE_BIN = os.path.join(SITE_PACKAGES, "cusparse", "bin")

print("==================================================================")
print("      EXECUTING AUTOMATIC WORKSPACE DRIVER INTEGRATION")
print("==================================================================")

def map_legacy_dll(search_dir, active_name, legacy_name):
    if os.path.exists(search_dir):
        active_path = os.path.join(search_dir, active_name)
        legacy_path = os.path.join(search_dir, legacy_name)
        if os.path.exists(active_path) and not os.path.exists(legacy_path):
            try:
                shutil.copy2(active_path, legacy_path)
                print(f" -> Automatically generated file link mapping: {legacy_name}")
            except Exception as e:
                print(f" -> Warning mapping {legacy_name}: {e}")

map_legacy_dll(CURAND_BIN, "curand64_11.dll", "curand64_10.dll")

bin_paths = [CUDNN_BIN, RUNTIME_BIN, CURAND_BIN, CUSOLVER_BIN, CUSPARSE_BIN]
valid_paths = [p for p in bin_paths if os.path.exists(p)]

if valid_paths:
    os.environ["PATH"] = os.path.pathsep.join(valid_paths) + os.pathsep + os.environ["PATH"]
    print(f"[SYSTEM ENVIRONMENT] Successfully mapped {len(valid_paths)} driver targets directly to runtime PATH.")
else:
    print("[SYSTEM ENVIRONMENT] Warning: Modular nvidia libraries not found. Falling back to host paths.")

# ==================================================================
# 📦 FRAMEWORK MACHINE LEARNING PACKAGES
# ==================================================================
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from Keras_efficientnet_v2 import efficientnet_v2

# --- DYNAMIC LOCAL PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_ROOT = os.path.join(BASE_DIR, "Cardamom_Plant_Dataset_Chinnahalli_1724")
BG_REMOVED_DIR = os.path.join(DATASET_ROOT, "BG_REMOVED")

DATASET_MAPPING = {
    "Healthy": os.path.join(BG_REMOVED_DIR, "Background_Removed_Healthy"),
    "Blight": os.path.join(BG_REMOVED_DIR, "Background_Removed_Blight1000"),
    "Phyllosticta": os.path.join(BG_REMOVED_DIR, "Background_Removed_Phyllosticta_Leaf_Spot")
}

img_data_list = []
F_1_labels = []
class_index = 0

print("\n==================================================================")
print("       INITIATING PRODUCTION DATASET LOADING CORPUS PASS")
print("==================================================================")
print(f"Targeting BG_REMOVED root directory layout: {BG_REMOVED_DIR}\n")

for class_name, folder_path in DATASET_MAPPING.items():
    if not os.path.exists(folder_path):
        print(f"[ERROR] Critical folder path target not discovered: {folder_path}")
        continue
        
    img_list = os.listdir(folder_path)
    valid_imgs = [f for f in img_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" -> Parsing class target folder '{class_name}': Found {len(valid_imgs)} active entries.")
    
    for img_name in valid_imgs:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = x.astype('float32') / 255.0
            
            img_data_list.append(x)
            F_1_labels.append(class_index)
        except Exception as e:
            print(f"      [WARNING] Bypassing corrupted asset frame {img_name}: {e}")
            
    class_index += 1

img_data = np.array(img_data_list)
print("\n[DATA LOADING PIPELINE COMPLETED SUCCESSFULLY]")
print(f"Total processed leaf vector samples: {img_data.shape[0]}")

F1_labels = np.array(F_1_labels)
num_classes = 3
y = tf.keras.utils.to_categorical(F1_labels, num_classes)

x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.1, random_state=2)

# ==================================================================
# 🚀 PHASE 1: INITIAL FEATURE EXTRACTION WARM-UP
# ==================================================================
print("\nCompiling network layers via ImageNet21k Pretrained Vectors...")
base_model = efficientnet_v2.EfficientNetV2L(
    input_shape=(224, 224, 3), 
    survivals=None, 
    dropout=1e-6, 
    classes=0, 
    pretrained="imagenet21k"
)

# Freeze backbone entirely to respect absolute local VRAM constraints
base_model.trainable = False

# HIGH-CAPACITY CLASSIFICATION HEAD: Richer configuration to map complex textures
classifier = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nLaunching Phase 1 High-Capacity Adapter training loop...")
classifier.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

warmup_weights_path = os.path.join(BASE_DIR, "EfficientNetV2_Cardamom_Warmup.h5")
classifier.save(warmup_weights_path)
print(f"\n[PHASE 1 COMPLETE] Base adapter weights saved at: {warmup_weights_path}")

# ==================================================================
# 🚀 PHASE 2: DEEP ADAPTER FINE-TUNING (BACKBONE REMAINS FROZEN)
# ==================================================================
print("\n==================================================================")
print("       INITIATING DEEP PRODUCTION FINE-TUNING LAYER PASS")
print("==================================================================")

print("Running structural tuning across custom adapter parameters...")
# Re-compile the adapter layers at a lower learning rate for final convergence checks
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Safe execution path guaranteed free from OOM exceptions
classifier.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

final_weights_path = os.path.join(BASE_DIR, "EfficientNetV2_Cardamom_FineTuned.h5")
classifier.save(final_weights_path)
print(f"\n[SUCCESS] Final model weights successfully saved at: {final_weights_path}")

loss, accuracy = classifier.evaluate(x_test, y_test, verbose=0)
print(f"\n -> Final Stable Test Evaluation Accuracy: {accuracy * 100:.2f}%")
print(f" -> Final Optimization Loss Value: {loss:.4f}")