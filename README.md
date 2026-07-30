# Cardamom Disease Detection

Deep-learning pipeline to classify cardamom leaf images into **Healthy**, **Blight**, and **Phyllosticta** using **EfficientNetV2-L**, with optional **Plant Disease Index (PDI)** severity scoring for diseased leaves.

Training uses the **background-removed (BG_REMOVED)** dataset only.

---

## What belongs on GitHub

Keep **source code and config** in the repo. Host **datasets**, **trained weights**, and **virtual environments** elsewhere (local disk, Git LFS, or GitHub Releases).

### Include in the repository

```text
Cardamom_Disease_Detection/
├── README.md
├── requirements.txt
├── requirements-train-gpu.txt      # optional GPU training extras
├── Dockerfile
├── .dockerignore
├── .gitignore
├── blight_engine.py                # PDI severity for Blight
├── phyllosticta_engine.py          # PDI severity for Phyllosticta
└── EfficientNetV2/
    ├── effv2.py                    # training script
    ├── test_blind_inference.py     # single-image inference
    ├── evaluate_full.py            # evaluate on BG_REMOVED dataset
    ├── evaluate_windows_pipeline.py
    ├── diagnose_model.py
    └── Keras_efficientnet_v2/
        └── efficientnet_v2.py        # EfficientNetV2 architecture
```

### Do **not** commit

| Item | Reason |
|------|--------|
| `tf_gpu_env/`, `venv310/`, `.venv/`, `.venv-1/` | Local virtual environments |
| `Cardamom_Plant_Dataset_Chinnahalli_1724/` | ~740 MB image dataset |
| `paper/` | Separate evaluation image set |
| `*.h5` model files | Each ~460 MB (exceeds GitHub 100 MB limit) |
| `phytograde.tar` | ~17 GB archive |
| `U-2-Net-master/` | Background-removal tool; dataset is already BG_REMOVED |
| `__pycache__/`, `.vs/`, `__MACOSX/` | Cache / IDE / macOS metadata |
| `EfficientNetV2/confusion_matrix.png` | Generated output |

### Optional files (legacy — skip unless you need them)

- `EfficientNetV2/effv2_legacy.py`
- `EfficientNetV2/utils.py`
- `EfficientNetV2/effv2.sh` (PBS cluster job script)

### Trained model weights

The fine-tuned classifier is **`EfficientNetV2_Cardamom_FineTuned.h5`** (~460 MB). Because it exceeds GitHub’s file-size limit, use one of:

1. **[Git LFS](https://git-lfs.com/)** — track `*.h5` in LFS
2. **GitHub Releases** — attach the `.h5` as a release asset
3. **Google Drive / Hugging Face** — link the download URL in this README

Place the downloaded weights in the **project root**:

```text
Cardamom_Disease_Detection/EfficientNetV2_Cardamom_FineTuned.h5
```

---

## Dataset (local only)

Download or copy the dataset separately. Training and `evaluate_full.py` expect this layout:

```text
Cardamom_Plant_Dataset_Chinnahalli_1724/
└── BG_REMOVED/
    ├── Background_Removed_Healthy/
    ├── Background_Removed_Blight1000/
    └── Background_Removed_Phyllosticta_Leaf_Spot/
```

| Class | Folder | Approx. images |
|-------|--------|----------------|
| Healthy | `Background_Removed_Healthy` | ~781 |
| Blight | `Background_Removed_Blight1000` | ~281 |
| Phyllosticta | `Background_Removed_Phyllosticta_Leaf_Spot` | ~663 |

Images are resized to **224×224** and normalized to `[0, 1]`. Only `.jpg`, `.jpeg`, and `.png` are loaded.

The original folders (`Blight1000`, `Healthy_1000`, etc.) are **not** used by the training script.

---

## Setup

**Python 3.10** is recommended.

```powershell
cd Cardamom_Disease_Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For **GPU training on Windows** (CUDA 11), create a dedicated env and install GPU wheels:

```powershell
python -m venv tf_gpu_env
tf_gpu_env\Scripts\activate
pip install -r requirements-train-gpu.txt
```

> `effv2.py` automatically adds NVIDIA DLL paths from `tf_gpu_env/Lib/site-packages/nvidia/` to `PATH`. Keep that folder name if you use the built-in GPU path logic, or activate the env before running training.

---

## Training

Two-phase training (warm-up + fine-tune) on the BG_REMOVED dataset:

```powershell
cd EfficientNetV2
python effv2.py
```

**Outputs** (written to project root):

| File | Description |
|------|-------------|
| `EfficientNetV2_Cardamom_Warmup.h5` | Phase 1 — frozen backbone |
| `EfficientNetV2_Cardamom_FineTuned.h5` | Final model for inference |

EfficientNetV2-L ImageNet21k weights are downloaded automatically on first run.

**Hyperparameters (in `effv2.py`):**

- Train/val split: 90% / 10% (`test_size=0.1`, `validation_split=0.1`)
- Phase 1: 15 epochs, LR `1e-3`, batch size 32
- Phase 2: 10 epochs, LR `1e-4`, batch size 32

---

## Inference

Requires `EfficientNetV2_Cardamom_FineTuned.h5` in the project root.

```powershell
cd EfficientNetV2
python test_blind_inference.py path\to\leaf_image.jpg
```

**Output:**

- Predicted class (Healthy / Blight / Phyllosticta)
- Softmax probabilities
- **PDI** (%) and severity tier for Blight and Phyllosticta

Severity tiers:

| PDI | Tier |
|-----|------|
| 0% | None (Healthy) |
| < 2% | Mild |
| 2–7% | Moderate |
| ≥ 7% | Severe |

---

## Evaluation

```powershell
cd EfficientNetV2

# Full pass on BG_REMOVED dataset
python evaluate_full.py

# Quick random sample per class
python diagnose_model.py

# Separate test set under paper/ (if available locally)
python evaluate_windows_pipeline.py
```

`evaluate_windows_pipeline.py` also needs `matplotlib` (`pip install matplotlib`).

---

## Docker (inference)

```powershell
docker build -t cardamom-disease .
docker run --rm cardamom-disease python test_blind_inference.py /app/sample.jpg
```

Mount your image and ensure `EfficientNetV2_Cardamom_FineTuned.h5` is copied into the image or mounted at `/app/`.

---

## Project architecture

```text
BG_REMOVED images
       │
       ▼
  effv2.py  ──►  EfficientNetV2-L + classifier head
       │
       ▼
EfficientNetV2_Cardamom_FineTuned.h5
       │
       ▼
test_blind_inference.py
       ├── Healthy  → PDI = 0
       ├── Blight   → blight_engine.py
       └── Phyllosticta → phyllosticta_engine.py
```

---

## Classes

| Index | Label | Severity engine |
|-------|-------|-----------------|
| 0 | Healthy | — |
| 1 | Blight | `blight_engine.py` |
| 2 | Phyllosticta | `phyllosticta_engine.py` |

---

## License

Add your license here before publishing.