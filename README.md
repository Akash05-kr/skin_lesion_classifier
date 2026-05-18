# VGGNet-16 Skin Lesion Classification on MILK10k (ISIC Multimodal)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/VGG16-Transfer%20Learning-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-11-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Medical%20AI-blueviolet?style=for-the-badge" />
</p>

> **Production-quality, single-file deep learning pipeline** — implements transfer learning with VGG16 on the MILK10k (ISIC Multimodal) dermoscopic dataset for 11-class skin lesion classification. Fully trained for 30 epochs, achieving **66.58% test accuracy** and **AUC-ROC of 0.8073**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Actual Training Results](#2-actual-training-results)
3. [Dataset Description](#3-dataset-description)
4. [Architecture](#4-architecture)
5. [Transfer Learning Strategy](#5-transfer-learning-strategy)
6. [Folder Structure](#6-folder-structure)
7. [Installation](#7-installation)
8. [Quick Start](#8-quick-start)
9. [Training Instructions](#9-training-instructions)
10. [Resuming Training](#10-resuming-training)
11. [Evaluation Instructions](#11-evaluation-instructions)
12. [Understanding the Outputs](#12-understanding-the-outputs)
13. [CLI Reference](#13-cli-reference)
14. [Hyperparameters](#14-hyperparameters)
15. [Metrics Explanation](#15-metrics-explanation)
16. [Reproducibility](#16-reproducibility)
17. [Known Limitations & Future Improvements](#17-known-limitations--future-improvements)
18. [Acknowledgements](#18-acknowledgements)

---

## 1. Project Overview

This project implements a **multi-class skin lesion classifier** using the **VGGNet-16** convolutional neural network architecture fine-tuned on the **MILK10k (ISIC Multimodal)** dermoscopic image dataset.

The entire pipeline — configuration, dataset loading, model building, training, validation, evaluation, checkpointing, and visualisation — is consolidated into a **single, self-contained `main.py`** file for simplicity and portability.

**Key Features:**

| Feature | Details |
|---|---|
| Model | VGG16 pretrained on ImageNet |
| Dataset | MILK10k (ISIC Multimodal), 5,240 dermoscopic images |
| Classes | 11 diagnostic categories |
| Split | Stratified 70 / 15 / 15 (train / val / test) |
| Training | 30 epochs, Adam optimizer, ReduceLROnPlateau LR scheduling |
| Early Stopping | Patience = 7 epochs on validation loss |
| Checkpointing | Best val-loss + final epoch checkpoints saved |
| Resume Support | `--resume` flag to continue from any saved checkpoint |
| AMP | Automatic Mixed Precision (auto-disabled on CPU) |
| Metrics | Accuracy, Precision, Recall, F1, AUC-ROC (OvR macro) |
| Visualisations | Training curves, confusion matrix heatmap, ROC curves |
| Reproducibility | Full seeding (Python, NumPy, PyTorch, cuDNN) |

---

## 2. Actual Training Results

The model was trained on a **CPU** for **30 full epochs** (~81 minutes total). The best checkpoint was saved at **Epoch 20** (lowest validation loss of **1.0521**).

### Per-Epoch Training Summary (Selected Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|-------|-----------|-----------|----------|---------|-----|
| 001 | 1.8695 | 48.54% | 1.3028 | 60.94% | 1e-4 |
| 005 | 1.2334 | 61.11% | 1.1708 | 63.74% | 1e-4 |
| 010 | 0.9786 | 68.02% | 1.0860 | 66.29% | 1e-4 |
| 015 | 0.8914 | 70.19% | 1.0879 | 65.01% | 1e-4 |
| 020 | 0.8325 | 71.72% | **1.0521** | **67.56%** | 1e-4 |
| 025 | 0.6237 | 78.89% | 1.2063 | 65.78% | 5e-5 |
| 030 | 0.5128 | 83.23% | 1.2776 | 66.28% | 5e-5 |

> Best model saved at **Epoch 20** — val_loss = 1.0521, val_acc = 67.56%.

---

### Test Set Results (787 images)

```
============================================================
  FINAL TEST RESULTS
============================================================
  Accuracy  : 0.6658  (66.58%)
  Precision : 0.6264
  Recall    : 0.6658
  F1-Score  : 0.6352
  AUC-ROC   : 0.8073
============================================================
```

### Validation Set Results (786 images)

```
  Accuracy  : 0.6756  (67.56%)
  Precision : 0.6357
  Recall    : 0.6756
  F1-Score  : 0.6427
  AUC-ROC   : 0.9061
```

### Per-Class Classification Report (Test Set)

```
              precision    recall  f1-score   support

       AKIEC       0.44      0.18      0.25        45
         BCC       0.72      0.91      0.81       379
     BEN_OTH       0.00      0.00      0.00         7
         BKL       0.41      0.32      0.36        82
          DF       0.00      0.00      0.00         8
         INF       0.00      0.00      0.00         7
     MAL_OTH       0.00      0.00      0.00         1
         MEL       0.59      0.44      0.50        68
          NV       0.72      0.71      0.71       112
       SCCKA       0.53      0.48      0.50        71
        VASC       1.00      0.43      0.60         7

    accuracy                           0.67       787
   macro avg       0.40      0.31      0.34       787
weighted avg       0.63      0.67      0.64       787
```

**Observations:**
- **BCC** (Basal Cell Carcinoma) dominates the dataset (379/787 test samples = 48%) and is predicted very well (F1 = 0.81).
- **NV** (Melanocytic Nevus) and **MEL** (Melanoma) show reasonable performance (F1 = 0.71 and 0.50).
- **Rare classes** (BEN_OTH, DF, INF, MAL_OTH) have near-zero support in the test set and are essentially unpredictable with the frozen-backbone approach due to severe class imbalance.
- The **AUC-ROC of 0.81** on the test set indicates the model has good discriminative ability across the 11 classes, well above the 0.5 random baseline.

---

## 3. Dataset Description

The **MILK10k** dataset is part of the ISIC (International Skin Imaging Collaboration) Multimodal challenge.

| Property | Details |
|---|---|
| Full Name | MILK10k — ISIC Multimodal Skin Lesion Classification |
| Total Lesions | 5,240 unique lesions |
| Image Modality Used | Dermoscopic images only |
| Total Images (dermoscopic) | 5,240 (one per lesion) |
| Classes | 11 diagnostic categories |
| Task | Supervised multi-class image classification |

### Class Distribution (Full Dataset)

| ID | Code | Full Name | Count |
|----|------|-----------|-------|
| 0 | AKIEC | Actinic Keratosis / Intraepithelial Carcinoma | 303 |
| 1 | BCC | Basal Cell Carcinoma | 2,522 |
| 2 | BEN_OTH | Benign Other | 44 |
| 3 | BKL | Benign Keratosis-like Lesion | 544 |
| 4 | DF | Dermatofibroma | 52 |
| 5 | INF | Infection | 50 |
| 6 | MAL_OTH | Malignant Other | 9 |
| 7 | MEL | Melanoma | 450 |
| 8 | NV | Melanocytic Nevus | 746 |
| 9 | SCCKA | Squamous Cell Carcinoma / Keratoacanthoma | 473 |
| 10 | VASC | Vascular Lesion | 47 |
| | **Total** | | **5,240** |

> **Note:** The dataset is heavily imbalanced — BCC alone accounts for ~48% of all samples.

### Dataset Placement

The MILK10k dataset must be organized as follows inside the project directory **before running**:

```
skin_lesion_classifier/data/dataset/
├── MILK10k_Training_Input/
│   ├── IL_0000001/
│   │   └── ISIC_xxxxxxx.jpg      <- dermoscopic image
│   ├── IL_0000002/
│   │   └── ISIC_xxxxxxx.jpg
│   └── ... (one folder per lesion_id)
├── MILK10k_Training_GroundTruth.csv
└── MILK10k_Training_Metadata.csv
```

The pipeline reads class labels from `MILK10k_Training_GroundTruth.csv` (one-hot encoded) and maps each `lesion_id` to its dermoscopic `isic_id` via `MILK10k_Training_Metadata.csv`.

---

## 4. Architecture

### VGGNet-16 Modified for 11-Class Classification

The original VGG16 classifier head (designed for 1000 ImageNet classes) is replaced with a custom head for 11-class output:

```
Input Image (224 x 224 x 3)
|
+-- Conv Block 1 -- 2x [Conv2d(64) -> ReLU] -> MaxPool2d  -->  112x112x64
+-- Conv Block 2 -- 2x [Conv2d(128) -> ReLU] -> MaxPool2d -->  56x56x128
+-- Conv Block 3 -- 3x [Conv2d(256) -> ReLU] -> MaxPool2d -->  28x28x256
+-- Conv Block 4 -- 3x [Conv2d(512) -> ReLU] -> MaxPool2d -->  14x14x512
+-- Conv Block 5 -- 3x [Conv2d(512) -> ReLU] -> MaxPool2d -->  7x7x512
|
+-- AdaptiveAvgPool2d(7x7) --------->  7x7x512 = 25,088 features
+-- Flatten
|
+-- FC1: Linear(25088 -> 4096) -> ReLU -> Dropout(0.5)
+-- FC2: Linear(4096 -> 4096)  -> ReLU -> Dropout(0.5)
+-- FC3: Linear(4096 -> 11)    [Raw logits for 11 classes]
```

| Parameter Group | Count |
|---|---|
| Total Parameters | ~134.3 million |
| Trainable (Phase 1 — FC head only) | ~119.6 million |
| Trainable (Phase 2 — all layers) | ~134.3 million |
| Non-trainable (frozen backbone) | ~14.7 million |

**Why VGG16?**
- Uniform 3×3 convolution architecture — easy to reason about and modify
- Deep enough (16 weight layers) to learn rich hierarchical visual features
- Excellent ImageNet pretrained weights available via torchvision
- Well-established baseline for skin lesion classification literature

---

## 5. Transfer Learning Strategy

Transfer learning adapts knowledge from a large general-purpose model (ImageNet) to a smaller, domain-specific task (skin lesion classification).

```
Phase 1 — Feature Extraction (default)
---------------------------------------
  ImageNet pretrained weights
           |
           v
  [FROZEN] VGG16 Backbone  (conv layers — not updated)
           |
           v
  [TRAINABLE] New FC Head  (trained from scratch on MILK10k)
           |
           v
  Output: 11-class softmax probabilities


Phase 2 — Fine-Tuning (optional, --fine_tune flag)
-----------------------------------------------------
  Best checkpoint from Phase 1
           |
           v
  [TRAINABLE] Entire VGG16   (very low LR = 1e-5)
           |
           v
  Output: 11-class softmax probabilities (domain-specialized)
```

**Benefits of this approach:**
- Phase 1 trains fast (only ~120M FC parameters update) and avoids overfitting the backbone to a small dataset
- Phase 2 (fine-tuning) can squeeze additional accuracy at the cost of longer training and higher risk of overfitting
- The `ReduceLROnPlateau` scheduler automatically halves the LR when validation loss plateaus, allowing smooth convergence

---

## 6. Folder Structure

```
skin_lesion_classifier/
|
+-- data/                                <- Place MILK10k dataset here
|   +-- dataset/
|       +-- MILK10k_Training_Input/      <- One folder per lesion_id
|       +-- MILK10k_Training_GroundTruth.csv
|       +-- MILK10k_Training_Metadata.csv
|
+-- results/                             <- Auto-created on first run
|   +-- plots/
|   |   +-- training_history.png         <- Loss & Accuracy curves
|   |   +-- test_confusion_matrix.png    <- Normalised confusion matrix
|   |   +-- test_roc_curves.png          <- Per-class + macro ROC curves
|   |   +-- val_confusion_matrix.png
|   |   +-- val_roc_curves.png
|   |
|   +-- metrics/
|   |   +-- test_metrics.json            <- All test-set metrics
|   |   +-- val_metrics.json             <- All val-set metrics
|   |
|   +-- models/
|       +-- vgg16_best_model.pth         <- Best val-loss checkpoint
|       +-- vgg16_final_model.pth        <- Final epoch checkpoint
|
+-- main.py                              <- Entire pipeline (single file)
+-- requirements.txt                     <- Python dependencies
+-- training.log                         <- Structured log (appended each run)
+-- training_console.log                 <- Full console output with progress bars
+-- README.md                            <- This file
+-- .gitignore
```

---

## 7. Installation

### Prerequisites

- Python 3.11+
- Anaconda or virtualenv (recommended)
- NVIDIA GPU with CUDA 11.8+ (strongly recommended; CPU is supported but slow)

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourusername/skin-lesion-vgg16.git
cd skin-lesion-vgg16/skin_lesion_classifier
```

### Step 2 — Create and activate a virtual environment

**Using Anaconda (recommended):**
```bash
conda create -n skin_lesion python=3.11
conda activate skin_lesion
```

**Using venv:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Install PyTorch with CUDA (GPU users only)

Visit https://pytorch.org/get-started/locally/ for your CUDA version. Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 5 — Place the dataset

Ensure the MILK10k dataset is placed as described in [Section 3](#3-dataset-description) before running.

---

## 8. Quick Start

```bash
# Full pipeline: train from scratch + evaluate on val & test sets
python main.py
```

This single command will:
1. Load and parse the MILK10k dataset from CSV ground truth
2. Perform stratified 70/15/15 train/val/test split
3. Build and freeze the VGG16 backbone, create a new 11-class head
4. Train for up to 30 epochs with early stopping (patience=7)
5. Save the best model checkpoint whenever val-loss improves
6. Evaluate on both validation and test sets
7. Save confusion matrices, ROC curves, training history plots to `results/plots/`
8. Save all metrics (accuracy, precision, recall, F1, AUC-ROC) to `results/metrics/`

---

## 9. Training Instructions

### Default training (Phase 1 — FC head only, recommended)

```bash
python main.py --mode train
```

### Custom hyperparameters

```bash
python main.py --mode train \
  --epochs 50 \
  --lr 0.0001 \
  --batch_size 16 \
  --weight_decay 0.0001
```

### Phase 2 Fine-Tuning (unfreeze all layers)

```bash
python main.py --mode train --fine_tune
```

> Use a lower learning rate (e.g., `--lr 0.00001`) when fine-tuning to avoid catastrophic forgetting.

### Train from scratch (no ImageNet weights)

```bash
python main.py --mode train --no_pretrained
```

### Disable AMP (for CPU or debugging)

```bash
python main.py --mode train --no_amp
```

### Monitor progress

Training logs are written to two files simultaneously:
- **`training.log`** — structured log with epoch summaries (human-readable, no progress bars)
- **`training_console.log`** — full console output including tqdm progress bars per batch

```bash
# Windows PowerShell — watch training.log in real-time
Get-Content training.log -Wait -Tail 20
```

---

## 10. Resuming Training

If training is interrupted, you can resume from the last best checkpoint without restarting from scratch.

### Resume from the best checkpoint

```bash
python main.py --resume
```

This will:
1. Load `results/models/vgg16_best_model.pth`
2. Restore both model weights and optimizer state
3. Continue training from `epoch = last_saved_epoch + 1`
4. Use the saved `best_val_loss` as the starting baseline for checkpoint saving

### Resume from a specific checkpoint

```bash
python main.py --resume --checkpoint vgg16_final_model.pth
```

> **Important:** The `--resume` flag requires a compatible checkpoint file to exist. If the file is not found, training will start from scratch with a warning.

---

## 11. Evaluation Instructions

### Evaluate the best saved model (val + test sets)

```bash
python main.py --mode eval
```

### Evaluate a specific checkpoint

```bash
python main.py --mode eval --checkpoint vgg16_final_model.pth
```

### Full pipeline (train then immediately evaluate)

```bash
python main.py --mode full
```

The evaluation pipeline:
1. Loads the specified checkpoint into the VGG16 model
2. Runs inference on the validation set — saves confusion matrix + ROC curves to `results/plots/val_*.png`
3. Runs inference on the test set — saves confusion matrix + ROC curves to `results/plots/test_*.png`
4. Saves all metrics to `results/metrics/val_metrics.json` and `results/metrics/test_metrics.json`
5. Prints a final summary to the console

---

## 12. Understanding the Outputs

### Training History Plot (`results/plots/training_history.png`)

Two side-by-side line charts showing **Loss** and **Accuracy** curves for train and validation sets across all epochs. Use this to diagnose:
- **Overfitting:** Training accuracy/loss keeps improving but validation diverges
- **Underfitting:** Both train and val curves plateau at poor performance
- **Good convergence:** Both curves decrease and stabilise together

### Confusion Matrix (`results/plots/test_confusion_matrix.png`)

A normalised N×N heatmap where each row sums to 1.0. Cell `[i, j]` shows the fraction of true class `i` samples predicted as class `j`. **Diagonal cells = correct predictions (recall per class).**

### ROC Curves (`results/plots/test_roc_curves.png`)

One-vs-Rest ROC curves for all 11 classes plus the macro average. The dotted diagonal line represents a random classifier (AUC=0.5). The further the curves bow toward the top-left, the better.

### Metrics JSON (`results/metrics/test_metrics.json`)

```json
{
    "accuracy": 0.6658,
    "precision_weighted": 0.6264,
    "recall_weighted": 0.6658,
    "f1_weighted": 0.6352,
    "auc_roc_macro_ovr": 0.8073,
    "classification_report": "...",
    "confusion_matrix": [[...], ...]
}
```

### Model Checkpoints (`results/models/`)

| File | Saved When |
|---|---|
| `vgg16_best_model.pth` | Every time val_loss improves (best model) |
| `vgg16_final_model.pth` | After the last training epoch (final state) |

Each `.pth` file contains: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_loss`, `val_acc`.

---

## 13. CLI Reference

```
usage: main.py [-h] [--mode {train,eval,full}]
               [--data_dir DATA_DIR]
               [--epochs EPOCHS] [--lr LR]
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY]
               [--seed SEED] [--fine_tune] [--no_pretrained]
               [--checkpoint CHECKPOINT] [--no_amp] [--resume]

Arguments:
  --mode           Pipeline mode: 'train' | 'eval' | 'full' (default: full)
  --data_dir       Root directory of the MILK10k dataset
  --epochs         Maximum training epochs (default: 30)
  --lr             Initial learning rate (default: 0.0001)
  --batch_size     Mini-batch size (default: 32)
  --weight_decay   Adam L2 regularisation coefficient (default: 0.0001)
  --seed           Random seed for full reproducibility (default: 42)
  --fine_tune      Unfreeze all backbone layers for end-to-end fine-tuning
  --no_pretrained  Train VGG16 from scratch without ImageNet weights
  --checkpoint     Checkpoint filename to load for eval or resume (default: vgg16_best_model.pth)
  --no_amp         Disable Automatic Mixed Precision (forced off on CPU anyway)
  --resume         Resume training from the checkpoint in --checkpoint
```

---

## 14. Hyperparameters

All default hyperparameters are defined at the top of `main.py` and can be overridden via CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | 30 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Initial Adam learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation coefficient |
| `FINE_TUNE_LR` | 1e-5 | LR used during Phase 2 fine-tuning |
| `BATCH_SIZE` | 32 | Images per mini-batch |
| `NUM_WORKERS` | 0 | DataLoader worker processes (0 = main thread, required on Windows) |
| `EARLY_STOPPING_PATIENCE` | 7 | Epochs without val-loss improvement before stopping |
| `LR_SCHEDULER_PATIENCE` | 3 | Epochs before halving the learning rate |
| `LR_SCHEDULER_FACTOR` | 0.5 | Multiplicative factor for LR reduction |
| `LR_SCHEDULER_MIN_LR` | 1e-7 | Minimum allowed learning rate |
| `IMAGE_SIZE` | (224, 224) | Resize target — required by VGG16 |
| `RANDOM_SEED` | 42 | Global random seed |
| `TRAIN_RATIO` | 0.70 | Fraction of data for training |
| `VAL_RATIO` | 0.15 | Fraction for validation |
| `TEST_RATIO` | 0.15 | Fraction for testing |
| `USE_AMP` | True | Automatic Mixed Precision (GPU only) |
| `FREEZE_FEATURES` | True | Freeze backbone in Phase 1 |

---

## 15. Metrics Explanation

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | Correct / Total | Quick overall measure; misleading on imbalanced datasets |
| **Precision (weighted)** | TP / (TP + FP), averaged by support | When false positives are costly |
| **Recall (weighted)** | TP / (TP + FN), averaged by support | When false negatives are costly (critical in medical diagnosis) |
| **F1-Score (weighted)** | 2 * (P * R) / (P + R), averaged by support | Best single metric for imbalanced multi-class problems |
| **AUC-ROC (macro OvR)** | Area under ROC curve, One-vs-Rest macro avg | Ranking quality; threshold-independent; 0.5 = random, 1.0 = perfect |

### AUC-ROC Interpretation Guide

```
0.90 - 1.00  -->  Excellent
0.80 - 0.90  -->  Good
0.70 - 0.80  -->  Fair
0.60 - 0.70  -->  Poor
0.50 - 0.60  -->  Fail (no better than random)
```

> This model achieved **AUC-ROC = 0.8073 on the test set** (Good) and **0.9061 on the validation set** (Excellent), demonstrating solid discriminative ability despite severe class imbalance.

---

## 16. Reproducibility

All experiments are fully reproducible. Seeds are fixed for every random source:

| Component | Method |
|---|---|
| Python `random` | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(42)` |
| cuDNN | `deterministic=True`, `benchmark=False` |
| OS hash seed | `PYTHONHASHSEED=42` |
| Data splitting | `train_test_split(..., random_state=42, stratify=labels)` |

To reproduce the exact results documented here:

```bash
python main.py --mode full --seed 42 --epochs 30 --lr 0.0001 --batch_size 32
```

> **Note:** Results may differ slightly between CPU and GPU runs due to floating-point non-determinism in certain CUDA operations.

---

## 17. Known Limitations & Future Improvements

### Current Limitations

| Issue | Impact |
|---|---|
| Severe class imbalance (BCC = 48%) | Rare classes (BEN_OTH, MAL_OTH, DF, INF, VASC) receive near-zero recall |
| CPU-only training | ~6-9 minutes per epoch; GPU would reduce this to ~30-60 seconds |
| Only dermoscopic modality used | The second paired clinical image is ignored (multimodal fusion opportunity) |
| Frozen backbone (Phase 1 only) | End-to-end fine-tuning may yield further accuracy gains |

### Recommended Improvements

| Area | Suggestion |
|---|---|
| **Class Imbalance** | Use `WeightedRandomSampler` or class-weighted `CrossEntropyLoss` |
| **Augmentation** | Add CutMix, MixUp, or elastic deformation for rare classes |
| **Architecture** | Try EfficientNet-B4, ResNet-50, or Vision Transformers |
| **Fine-Tuning** | Enable `--fine_tune` after Phase 1 convergence for domain specialisation |
| **Multimodal** | Fuse dermoscopic + clinical images via dual-branch network |
| **Explainability** | Add Grad-CAM visualisations to highlight discriminative regions |
| **Hyperparameter Search** | Use Optuna for automated hyperparameter optimisation |
| **Cross-Validation** | Stratified k-fold for more statistically robust performance estimates |
| **Deployment** | Export to ONNX and serve via FastAPI for clinical use |

---

## 18. Acknowledgements

- **ISIC Archive** — for the MILK10k dermoscopic image collection and challenge
- **Simonyan & Zisserman (2015)** — *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* — the original VGGNet paper
- **PyTorch & torchvision** teams — for pretrained VGG16 weights and the deep learning framework
- **scikit-learn** — for stratified splitting, metrics, and confusion matrix utilities

---

<p align="center">
  Built for medical AI research and education.<br/>
  <strong>Accuracy: 66.58% | AUC-ROC: 0.8073 | 30 Epochs | 5,240 Dermoscopic Images | 11 Classes</strong>
</p>
