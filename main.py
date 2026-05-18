

# ==============================================================================
# src/py
# ==============================================================================
"""
py — Central Configuration Module
=========================================
All hyperparameters, paths, and settings are defined here.
Centralising configuration makes the project easy to tune and reproduce.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"          # Place your MILK10k dataset here
RESULTS_DIR: Path = ROOT_DIR / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
MODELS_DIR: Path = RESULTS_DIR / "models"

# ─────────────────────────────────────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# 11 diagnostic classes in MILK10k (column order from GroundTruth CSV)
CLASS_NAMES: list[str] = [
    "AKIEC",     # Actinic Keratosis / Intraepithelial Carcinoma
    "BCC",       # Basal Cell Carcinoma
    "BEN_OTH",   # Benign Other
    "BKL",       # Benign Keratosis-like Lesion
    "DF",        # Dermatofibroma
    "INF",       # Infection
    "MAL_OTH",   # Malignant Other
    "MEL",       # Melanoma
    "NV",        # Melanocytic Nevus
    "SCCKA",     # Squamous Cell Carcinoma / Keratoacanthoma
    "VASC",      # Vascular Lesion
]
NUM_CLASSES: int = len(CLASS_NAMES)


# Stratified split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE: tuple[int, int] = (224, 224)   # VGG16 expects 224×224
# ImageNet mean and std for normalisation (standard for pretrained models)
MEAN: list[float] = [0.485, 0.456, 0.406]
STD:  list[float] = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION (training only)
# ─────────────────────────────────────────────────────────────────────────────
USE_AUGMENTATION: bool = True
ROTATION_DEGREES: int  = 15          # ±15° random rotation
COLOR_JITTER: dict = {               # Brightness / contrast / saturation
    "brightness": 0.2,
    "contrast":   0.2,
    "saturation": 0.2,
    "hue":        0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE: int    = 32
NUM_WORKERS: int   = 0        # 0 = main process only (required on Windows)
PIN_MEMORY: bool   = False    # Only useful with CUDA; set False for CPU

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
PRETRAINED: bool         = True   # Use ImageNet pretrained weights
FREEZE_FEATURES: bool    = True   # Freeze conv layers in Phase-1 training
FINE_TUNE: bool          = False  # Set True to unfreeze all layers (Phase-2)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
NUM_EPOCHS: int         = 30
LEARNING_RATE: float    = 1e-4
WEIGHT_DECAY: float     = 1e-4    # L2 regularisation
FINE_TUNE_LR: float     = 1e-5   # Lower LR when fine-tuning all layers

# Early Stopping
EARLY_STOPPING_PATIENCE: int = 7  # Stop if val-loss doesn't improve for N epochs

# Learning Rate Scheduler (ReduceLROnPlateau)
LR_SCHEDULER_PATIENCE: int   = 3
LR_SCHEDULER_FACTOR:   float = 0.5
LR_SCHEDULER_MIN_LR:   float = 1e-7

# Mixed Precision (AMP) — automatically enabled when CUDA is available
USE_AMP: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE: Path = ROOT_DIR / "training.log"

# ─────────────────────────────────────────────────────────────────────────────
# SAVED MODEL NAMES
# ─────────────────────────────────────────────────────────────────────────────
BEST_MODEL_NAME:  str = "vgg16_best_model.pth"
FINAL_MODEL_NAME: str = "vgg16_final_model.pth"


def ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    for d in (DATA_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)




# ==============================================================================
# src/utils.py
# ==============================================================================
"""
utils.py — Shared Utility Functions
=====================================
Provides:
  • Random-seed initialisation for full reproducibility
  • Logging setup (file + console)
  • EarlyStopping callback
  • Metric computation helpers
  • Training-history plotting
  • Confusion-matrix and ROC-curve visualisations
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize



# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Fix all random seeds so that every run produces identical results.

    Seeds:
      - Python's built-in random module
      - NumPy
      - PyTorch CPU and GPU RNGs
      - CUDA deterministic algorithms flag
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                  # Multi-GPU
    # Force cuDNN to use deterministic algorithms (slight speed trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"[Seed] All random seeds set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """
    Configure root logger to write to both the console and a log file.

    Args:
        log_file: Path where the log file will be written.

    Returns:
        Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"[Logging] Log file: {log_file}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority: CUDA GPU → MPS (Apple Silicon) → CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("[Device] Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logging.info("[Device] CPU (no GPU detected)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when the validation loss stops improving.

    Attributes:
        patience  : Number of epochs to wait before stopping.
        delta     : Minimum change to qualify as an improvement.
        best_score: Best validation loss seen so far (stored as negative).
        counter   : Epochs without improvement.
        stop      : Boolean flag — set True when training should halt.
    """

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        self.patience   = patience
        self.delta      = delta
        self.verbose    = verbose
        self.best_score: Optional[float] = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float) -> None:
        """
        Call at the end of each epoch with the current validation loss.

        Args:
            val_loss: Validation loss for the current epoch.
        """
        score = -val_loss   # Convert minimisation to maximisation

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            # No meaningful improvement
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.stop = True
                logging.info("[EarlyStopping] Triggered — halting training.")

        else:
            # Improvement detected
            self.best_score = score
            self.counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true      : Ground-truth integer labels (N,).
        y_pred      : Predicted integer labels (N,).
        y_prob      : Predicted class probabilities (N, C) — output of softmax.
        class_names : Human-readable class labels.

    Returns:
        Dictionary containing all computed metrics:
        accuracy, precision, recall, f1, auc_roc, classification_report,
        confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    n_classes = len(class_names)

    # Basic metrics
    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # AUC-ROC (One-vs-Rest, macro average)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        auc_roc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        # Can happen if a class has no positive samples in the split
        auc_roc = float("nan")
        logging.warning("[Metrics] AUC-ROC computation failed (missing class samples).")

    # Detailed classification report
    cls_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy":               accuracy,
        "precision_weighted":     precision,
        "recall_weighted":        recall,
        "f1_weighted":            f1,
        "auc_roc_macro_ovr":      auc_roc,
        "classification_report":  cls_report,
        "confusion_matrix":       cm,
    }

    logging.info(
        f"[Metrics] Acc={accuracy:.4f} | P={precision:.4f} | R={recall:.4f} | "
        f"F1={f1:.4f} | AUC={auc_roc:.4f}"
    )
    return metrics


def save_metrics(metrics: Dict[str, Any], filename: str = "test_metrics.json") -> None:
    """
    Persist computed metrics to a JSON file inside results/metrics/.

    Args:
        metrics : Dictionary of computed metrics.
        filename: Output filename (default: test_metrics.json).
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = METRICS_DIR / filename

    # classification_report is a string — keep as-is; cm is already a list
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=str)

    logging.info(f"[Metrics] Saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION — Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List[float]],
    save: bool = True,
) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        history : Dictionary with keys:
                  'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save    : If True, saves the figure to results/plots/.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VGG16 Training History — MILK10k", fontsize=15, fontweight="bold")

    # ── Loss curves ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # ── Accuracy curves ───────────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "training_history.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"[Plot] Training history saved → {path}")

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save: bool = True,
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Generate a normalised confusion-matrix heatmap with seaborn.

    Normalisation divides each row by the row sum so that each cell shows the
    fraction of true instances classified into each predicted class.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : List of class name strings.
        save        : Save figure if True.
        filename    : Output filename.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(10, n)          # Scale figure to number of classes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Normalised Confusion Matrix — VGG16 on MILK10k", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"[Plot] Confusion matrix saved → {path}")

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save: bool = True,
    filename: str = "roc_curves.png",
) -> None:
    """
    Plot One-vs-Rest ROC curves for every class plus the macro-average.

    Args:
        y_true      : Ground-truth integer labels (N,).
        y_prob      : Softmax probabilities (N, C).
        class_names : List of class name strings.
        save        : Save figure if True.
        filename    : Output filename.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_dict: Dict[int, np.ndarray] = {}
    tpr_dict: Dict[int, np.ndarray] = {}
    auc_dict: Dict[int, float] = {}

    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_dict[i] = float(auc(fpr_dict[i], tpr_dict[i]))

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = float(auc(all_fpr, mean_tpr))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_classes)

    for i in range(n_classes):
        ax.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=cmap(i),
            lw=1.5,
            label=f"{class_names[i]} (AUC={auc_dict[i]:.2f})",
        )

    ax.plot(
        all_fpr,
        mean_tpr,
        "k--",
        lw=2.5,
        label=f"Macro Average (AUC={macro_auc:.2f})",
    )
    ax.plot([0, 1], [0, 1], "grey", lw=1, linestyle=":")  # Diagonal reference
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("One-vs-Rest ROC Curves — VGG16 on MILK10k", fontsize=13)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"[Plot] ROC curves saved → {path}")

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Model Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filename: str = BEST_MODEL_NAME,
) -> None:
    """
    Save model weights, optimizer state, and training metadata.

    Saving the optimizer state enables resuming training exactly where it left off.

    Args:
        model    : PyTorch model.
        optimizer: Optimizer instance.
        epoch    : Current epoch number.
        val_loss : Validation loss at this checkpoint.
        val_acc  : Validation accuracy at this checkpoint.
        filename : Checkpoint filename inside results/models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    torch.save(
        {
            "epoch":      epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        },
        path,
    )
    logging.info(f"[Checkpoint] Saved → {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filename: str = BEST_MODEL_NAME,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load a previously saved model checkpoint.

    Args:
        model    : PyTorch model (must have identical architecture).
        optimizer: Optimizer instance (can be None for inference).
        filename : Checkpoint filename inside results/models/.
        device   : Target device (inferred if None).

    Returns:
        Tuple of (model, optimizer, start_epoch, best_val_loss).
    """
    if device is None:
        device = get_device()

    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch    = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    logging.info(f"[Checkpoint] Loaded ← {path}  (epoch={epoch}, val_loss={val_loss:.4f})")

    return model, optimizer, epoch, val_loss




# ==============================================================================
# src/dataset.py
# ==============================================================================
"""
dataset.py — MILK10k Dataset Loading and Preprocessing
========================================================
Handles the actual MILK10k (ISIC Multimodal) dataset structure:

  data/dataset/
  ├── MILK10k_Training_Input/
  │   ├── IL_0000652/
  │   │   ├── ISIC_4671410.jpg   ← dermoscopic image
  │   │   └── ISIC_8149219.jpg   ← clinical close-up image
  │   └── ...
  ├── MILK10k_Training_GroundTruth.csv  ← one-hot class labels per lesion
  ├── MILK10k_Training_Metadata.csv     ← maps lesion_id → isic_id → image_type
  └── ...

Strategy:
  • Use the DERMOSCOPIC image per lesion (standard for skin lesion classification)
  • Parse GroundTruth CSV to get the integer class label from the one-hot columns
  • Ignore lesions that have ambiguous labels (no class marked = 1)
  • Perform stratified 70/15/15 train/val/test split at the LESION level
    (both images of a lesion always end up in the same split)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR: Path = DATA_DIR / "dataset"
TRAIN_INPUT_DIR: Path  = DATASET_DIR / "MILK10k_Training_Input"
GROUNDTRUTH_CSV: Path  = DATASET_DIR / "MILK10k_Training_GroundTruth.csv"
METADATA_CSV: Path     = DATASET_DIR / "MILK10k_Training_Metadata.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CSV-BASED DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_dataset(
    data_dir: Path = DATA_DIR,
    image_modality: str = "dermoscopic",
) -> Tuple[List[str], List[int], List[str]]:
    """
    Build image path and label lists from the MILK10k CSV ground truth.

    Steps:
      1. Read GroundTruth CSV → derive integer class index per lesion via argmax
         of the one-hot columns.
      2. Read Metadata CSV → build a mapping lesion_id → isic_id for the
         chosen modality ('dermoscopic' by default).
      3. Resolve the image file path for each lesion.
      4. Skip lesions whose folder or file is missing.

    Args:
        data_dir      : Project data root (unused directly, kept for API compat).
        image_modality: 'dermoscopic' or 'clinical: close-up'.

    Returns:
        image_paths : List of absolute path strings.
        labels      : Corresponding integer class indices.
        class_names : Ordered class names (matching column order in CSV).
    """
    # ── Validate paths ────────────────────────────────────────────────────────
    for p in (TRAIN_INPUT_DIR, GROUNDTRUTH_CSV, METADATA_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file/directory not found: {p}\n"
                "Please place the MILK10k dataset inside data/dataset/"
            )

    # ── 1. Ground truth → integer labels ─────────────────────────────────────
    gt = pd.read_csv(GROUNDTRUTH_CSV)
    class_cols = [c for c in gt.columns if c != "lesion_id"]  # 11 class columns
    class_names = class_cols                                   # e.g. ['AKIEC', 'BCC', …]

    # argmax of one-hot row → integer class index
    gt["label"] = gt[class_cols].values.argmax(axis=1)

    # Keep only rows where exactly one class is marked (valid labels)
    gt = gt[gt[class_cols].sum(axis=1) == 1].copy()
    logging.info(f"[Dataset] Ground truth loaded: {len(gt)} valid lesions")

    # ── 2. Metadata → isic_id for chosen modality ─────────────────────────────
    meta = pd.read_csv(METADATA_CSV)
    # Filter to desired image type
    meta_mod = meta[meta["image_type"] == image_modality][["lesion_id", "isic_id"]]
    # Build dict: lesion_id → isic_id
    lesion_to_isic: Dict[str, str] = dict(
        zip(meta_mod["lesion_id"], meta_mod["isic_id"])
    )
    logging.info(
        f"[Dataset] Metadata filtered to '{image_modality}': "
        f"{len(lesion_to_isic)} records"
    )

    # ── 3. Resolve file paths ─────────────────────────────────────────────────
    image_paths: List[str] = []
    labels: List[int]      = []
    skipped = 0

    for _, row in gt.iterrows():
        lesion_id = row["lesion_id"]
        label     = int(row["label"])

        # Look up the isic_id for this modality
        isic_id = lesion_to_isic.get(lesion_id)
        if isic_id is None:
            skipped += 1
            continue

        # Build path: MILK10k_Training_Input/<lesion_id>/<isic_id>.jpg
        img_path = TRAIN_INPUT_DIR / lesion_id / f"{isic_id}.jpg"
        if not img_path.exists():
            # Try .png
            img_path = img_path.with_suffix(".png")
            if not img_path.exists():
                skipped += 1
                continue

        image_paths.append(str(img_path))
        labels.append(label)

    logging.info(
        f"[Dataset] Resolved {len(image_paths)} images "
        f"({skipped} skipped — missing file or metadata)"
    )

    if not image_paths:
        raise ValueError(
            "No images could be resolved. Check that MILK10k_Training_Input "
            "is present and metadata CSV matches the folder contents."
        )

    # Log class distribution
    from collections import Counter
    dist = Counter(labels)
    for idx, name in enumerate(class_names):
        logging.info(f"  Class [{idx:02d}] {name}: {dist.get(idx, 0)} images")

    return image_paths, labels, class_names


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT  (identical API to original — no changes needed upstream)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = RANDOM_SEED,
) -> Tuple[
    List[str], List[str], List[str],
    List[int], List[int], List[int],
]:
    """
    Stratified 70 / 15 / 15 train / val / test split.

    Two-stage strategy:
      Stage 1: all → (trainval 85%) | (test 15%)
      Stage 2: trainval → (train 70/85%) | (val 15/85%)

    The stratify parameter ensures each split contains the same class
    proportions as the full dataset.
    """
    import numpy as np
    labels_arr = np.array(labels)

    test_size = 1.0 - train_ratio - val_ratio          # 0.15
    X_tv, X_test, y_tv, y_test = train_test_split(
        image_paths, labels_arr,
        test_size=test_size,
        stratify=labels_arr,
        random_state=seed,
    )

    val_rel = val_ratio / (train_ratio + val_ratio)    # ≈ 0.1765
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_rel,
        stratify=y_tv,
        random_state=seed,
    )

    logging.info(
        f"[Split] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}"
    )
    return (
        list(X_train), list(X_val), list(X_test),
        list(y_train), list(y_val), list(y_test),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Build torchvision transform pipelines for each data split.

    Training augmentations reduce over-fitting by synthetically increasing
    dataset diversity without requiring additional labelled data.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        Composed torchvision transform pipeline.
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    if split == "train" and USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.ColorJitter(**COLOR_JITTER),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for MILK10k skin lesion classification.

    Loads images on-the-fly from disk (memory-efficient for 10k+ images).
    Applies the supplied transform pipeline before returning tensors.

    Args:
        image_paths : List of absolute paths to image files.
        labels      : Corresponding integer class indices.
        transform   : torchvision transform to apply (can be None).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
    ) -> None:
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Load image from disk, apply transform, return (tensor, label)."""
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {img_path}: {exc}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: Path = DATA_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    End-to-end function: discover → split → wrap → return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # 1. Discover
    image_paths, labels, class_names = discover_dataset(data_dir)

    # 2. Split
    (
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
    ) = stratified_split(image_paths, labels)

    # 3. Datasets
    train_ds = SkinLesionDataset(train_paths, train_labels, get_transforms("train"))
    val_ds   = SkinLesionDataset(val_paths,   val_labels,   get_transforms("val"))
    test_ds  = SkinLesionDataset(test_paths,  test_labels,  get_transforms("test"))

    # 4. DataLoaders
    kw = dict(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    logging.info(
        f"[DataLoaders] Train={len(train_loader)} batches | "
        f"Val={len(val_loader)} batches | Test={len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader, class_names




# ==============================================================================
# src/model.py
# ==============================================================================
"""
model.py — VGGNet-16 Model Definition
=======================================
Implements the VGG16 architecture with ImageNet pretrained weights
via transfer learning for multi-class skin lesion classification.

VGG16 ARCHITECTURE OVERVIEW
────────────────────────────
Original paper: "Very Deep Convolutional Networks for Large-Scale Image
Recognition" (Simonyan & Zisserman, ICLR 2015).

Input: 224×224×3
│
├── Block 1: 2 × Conv(64) → MaxPool  → 112×112×64
├── Block 2: 2 × Conv(128) → MaxPool → 56×56×128
├── Block 3: 3 × Conv(256) → MaxPool → 28×28×256
├── Block 4: 3 × Conv(512) → MaxPool → 14×14×512
├── Block 5: 3 × Conv(512) → MaxPool → 7×7×512
│
├── Flatten → 25088
├── FC(4096) → ReLU → Dropout(0.5)
├── FC(4096) → ReLU → Dropout(0.5)
└── FC(NUM_CLASSES) → output logits

TRANSFER LEARNING STRATEGY
───────────────────────────
Phase 1 (feature extraction):
  • Load ImageNet pretrained weights.
  • Freeze all convolutional layers.
  • Only the new classification head (FC layers) is trained.
  • Faster convergence; prevents catastrophic forgetting on small datasets.

Phase 2 (fine-tuning, optional):
  • Unfreeze all layers.
  • Train end-to-end with a very small learning rate (1e-5).
  • Allows the network to specialise its feature representations.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights



# ─────────────────────────────────────────────────────────────────────────────
# VGG16 MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class VGG16Classifier(nn.Module):
    """
    VGG16-based classifier for skin lesion diagnosis.

    The convolutional backbone is taken directly from the pretrained torchvision
    implementation; only the final classification head is replaced to output
    predictions for `num_classes` categories.

    Attributes:
        features   : VGG16 convolutional feature extractor (5 conv blocks).
        avgpool    : Adaptive average pooling → fixed 7×7 spatial output.
        classifier : Custom fully-connected classification head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained:  bool = PRETRAINED,
        freeze_features: bool = FREEZE_FEATURES,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialise the VGG16 classifier.

        Args:
            num_classes     : Number of output classes.
            pretrained      : Load ImageNet pretrained weights.
            freeze_features : Freeze the convolutional backbone.
            dropout         : Dropout probability in the FC head.
        """
        super().__init__()

        # ── Load pretrained VGG16 backbone ───────────────────────────────────
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16   = models.vgg16(weights=weights)
        logging.info(
            f"[Model] VGG16 loaded (pretrained={pretrained})"
        )

        # ── Reuse the convolutional feature extractor as-is ──────────────────
        # features: Sequential of Conv2d + ReLU + MaxPool layers
        self.features = vgg16.features

        # Adaptive pool keeps spatial size at 7×7 regardless of input size
        self.avgpool = vgg16.avgpool

        # ── Replace the original 1000-class head with a custom one ────────────
        # Original VGG16 head: 4096 → 4096 → 1000
        # Our head          : 4096 → 4096 → num_classes
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(512 * 7 * 7, 4096),   # 25088 → 4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Layer 2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Output layer (raw logits — no Softmax; handled by CrossEntropyLoss)
            nn.Linear(4096, num_classes),
        )

        # Initialise the new head with Xavier uniform for stable training
        self._init_weights()

        # ── Optionally freeze the backbone ───────────────────────────────────
        if freeze_features:
            self.freeze_backbone()

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Apply Xavier uniform initialisation to all Linear layers in the
        classification head. This produces better gradient flow than the
        default random initialisation.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Freeze / Unfreeze ────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all convolutional layers so only the FC head is updated.
        Used during Phase 1 (feature extraction mode).
        """
        for param in self.features.parameters():
            param.requires_grad = False
        logging.info("[Model] Backbone FROZEN — training FC head only.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all layers to enable end-to-end fine-tuning.
        Call this to enter Phase 2 with a reduced learning rate.
        """
        for param in self.features.parameters():
            param.requires_grad = True
        logging.info("[Model] Backbone UNFROZEN — full fine-tuning enabled.")

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full network.

        Args:
            x: Input tensor of shape (B, C, H, W) — expects 224×224 images.

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        x = self.features(x)    # Convolutional feature maps
        x = self.avgpool(x)     # → (B, 512, 7, 7)
        x = torch.flatten(x, 1) # → (B, 25088)
        x = self.classifier(x)  # → (B, num_classes)
        return x

    # ── Utility ──────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """
        Count trainable and total parameters.

        Returns:
            Dict with 'trainable' and 'total' parameter counts.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes:     int  = NUM_CLASSES,
    pretrained:      bool = PRETRAINED,
    freeze_features: bool = FREEZE_FEATURES,
    device: Optional[torch.device] = None,
) -> VGG16Classifier:
    """
    Construct and return a VGG16Classifier moved to the target device.

    Args:
        num_classes     : Number of output classes.
        pretrained      : Use ImageNet pretrained weights.
        freeze_features : Freeze convolutional backbone.
        device          : Compute device; auto-detected if None.

    Returns:
        VGG16Classifier instance on the target device.
    """
    if device is None:
        from src.utils import get_device
        device = get_device()

    model = VGG16Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_features=freeze_features,
    )
    model = model.to(device)

    params = model.count_parameters()
    logging.info(
        f"[Model] Parameters — Trainable: {params['trainable']:,} | "
        f"Total: {params['total']:,}"
    )
    return model




# ==============================================================================
# src/train.py
# ==============================================================================
"""
train.py — Training and Validation Pipeline
=============================================
Implements:
  • One-epoch training function (with optional AMP / mixed precision)
  • One-epoch validation function
  • Full multi-epoch training loop with:
      – Best-model saving
      – Early stopping
      – ReduceLROnPlateau scheduler
      – Training history accumulation
      – tqdm progress bars
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm



# ─────────────────────────────────────────────────────────────────────────────
# ONE-EPOCH TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device:    torch.device,
    scaler:    Optional[GradScaler] = None,
    epoch:     int = 0,
) -> Tuple[float, float]:
    """
    Run one complete pass through the training DataLoader.

    Mixed Precision (AMP):
      • autocast wraps the forward pass, using float16 for eligible ops.
      • GradScaler scales gradients to prevent underflow in float16.
      • Falls back to standard float32 if scaler is None (CPU or MPS).

    Args:
        model     : Model in train() mode.
        loader    : Training DataLoader.
        criterion : Loss function (CrossEntropyLoss).
        optimizer : Optimiser instance.
        device    : Compute device.
        scaler    : GradScaler for AMP (None → no AMP).
        epoch     : Current epoch number (for progress bar label).

    Returns:
        (mean_loss, accuracy_percent): Average loss and accuracy over the epoch.
    """
    model.train()

    running_loss   = 0.0
    correct        = 0
    total_samples  = 0

    pbar = tqdm(
        loader,
        desc=f"  Epoch {epoch:03d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # Slightly faster than zero_grad()

        # ── Forward pass (with optional AMP) ─────────────────────────────────
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping inside AMP (prevents exploding gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Accumulate stats ─────────────────────────────────────────────────
        batch_size    = labels.size(0)
        running_loss += loss.item() * batch_size

        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        # Live progress bar update
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total_samples * 100:.2f}%",
        )

    pbar.close()

    mean_loss = running_loss / total_samples
    accuracy  = correct / total_samples * 100.0
    return mean_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# ONE-EPOCH VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int = 0,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation (or test) DataLoader for one epoch.

    @torch.no_grad() disables gradient tracking for speed and memory savings.

    Args:
        model     : Model in eval() mode.
        loader    : Validation / test DataLoader.
        criterion : Loss function.
        device    : Compute device.
        epoch     : Current epoch number (for progress bar label).

    Returns:
        (mean_loss, accuracy_percent)
    """
    model.eval()

    running_loss  = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"  Epoch {epoch:03d} [Val]  ",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        batch_size    = labels.size(0)
        running_loss += loss.item() * batch_size

        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total_samples * 100:.2f}%",
        )

    pbar.close()

    mean_loss = running_loss / total_samples
    accuracy  = correct / total_samples * 100.0
    return mean_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# FULL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    num_epochs:   int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    use_amp:      bool  = USE_AMP,
    resume_checkpoint: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    End-to-end training routine with validation, scheduling, and checkpointing.

    Algorithm:
      1. Create Adam optimiser (only over trainable parameters).
      2. ReduceLROnPlateau scheduler — halves LR after N val-loss plateaus.
      3. GradScaler for AMP (disabled on CPU / MPS).
      4. Loop over epochs:
         a. Train one epoch → record loss & acc.
         b. Validate one epoch → record loss & acc.
         c. Step scheduler with val_loss.
         d. Save best checkpoint (lowest val_loss).
         e. Check early stopping criterion.
      5. Return full training history dict.

    Args:
        model        : VGG16Classifier (or any nn.Module).
        train_loader : Training DataLoader.
        val_loader   : Validation DataLoader.
        device       : Compute device.
        num_epochs   : Maximum number of epochs.
        lr           : Initial learning rate.
        weight_decay : L2 regularisation coefficient.
        use_amp      : Enable Automatic Mixed Precision.

    Returns:
        history: Dict with keys 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' — each a list of per-epoch floats.
    """
    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Only optimise parameters that require gradients (non-frozen layers)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    logging.info(
        f"[Optimizer] Adam | LR={lr} | WeightDecay={weight_decay}"
    )

    # ── Loss Function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss combines LogSoftmax + NLLLoss; expects raw logits
    criterion = nn.CrossEntropyLoss()

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",                         # Minimise val_loss
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    # AMP only works with CUDA; disable on CPU / MPS
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler() if amp_enabled else None
    if amp_enabled:
        logging.info("[AMP] Mixed Precision Training ENABLED")
    else:
        logging.info("[AMP] Mixed Precision Training DISABLED")

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # ── History Tracking ──────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")
    start_epoch = 1

    if resume_checkpoint:
        try:
            model, optimizer, loaded_epoch, loaded_val_loss = load_checkpoint(
                model, optimizer, filename=resume_checkpoint, device=device
            )
            start_epoch = loaded_epoch + 1
            best_val_loss = loaded_val_loss
        except FileNotFoundError:
            logging.warning(f"Checkpoint {resume_checkpoint} not found. Starting from scratch.")

    training_start = time.time()

    # ═════════════════════════════════════════════════════════════════════════
    # EPOCH LOOP
    # ═════════════════════════════════════════════════════════════════════════
    logging.info(f"[Training] Starting — {num_epochs} epochs max")
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # ── Scheduler step ────────────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Record history ────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.2f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}% | "
            f"LR={current_lr:.2e} | Time={epoch_time:.1f}s"
        )

        # ── Save best model ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                filename=BEST_MODEL_NAME,
            )

        # ── Early stopping check ──────────────────────────────────────────────
        early_stopping(val_loss)
        if early_stopping.stop:
            logging.info(f"[Training] Early stopping at epoch {epoch}")
            break

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, epoch, val_loss, val_acc,
        filename=FINAL_MODEL_NAME,
    )

    total_time = time.time() - training_start
    logging.info(
        f"[Training] Complete — Total time: {total_time / 60:.1f} min | "
        f"Best ValLoss: {best_val_loss:.4f}"
    )

    return history




# ==============================================================================
# src/evaluate.py
# ==============================================================================
"""
evaluate.py — Test-Set Evaluation Pipeline
============================================
Runs inference on the held-out test set and computes:
  • Accuracy, Precision, Recall, F1-Score
  • Macro AUC-ROC (One-vs-Rest)
  • Confusion matrix
  • Per-class classification report
  • All visualisation plots (confusion matrix, ROC curves)
  • Saves metrics to JSON
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm




# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — COLLECT PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over an entire DataLoader and collect predictions.

    Uses @torch.no_grad() to skip gradient tracking (inference only).

    Args:
        model  : Trained model in eval() mode.
        loader : DataLoader to evaluate (val or test).
        device : Compute device.

    Returns:
        y_true : Ground-truth integer labels (N,).
        y_pred : Predicted integer labels (N,).
        y_prob : Softmax probabilities (N, C).
    """
    model.eval()

    all_labels: List[np.ndarray] = []
    all_preds:  List[np.ndarray] = []
    all_probs:  List[np.ndarray] = []

    pbar = tqdm(loader, desc="  [Evaluate] Inference", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass → raw logits
        logits = model(images)                          # (B, C)
        probs  = torch.softmax(logits, dim=1)           # (B, C) probabilities
        preds  = logits.argmax(dim=1)                   # (B,)  predicted class

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    pbar.close()

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds,  axis=0)
    y_prob = np.concatenate(all_probs,  axis=0)

    return y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
    split_name:  str = "test",
) -> Dict:
    """
    End-to-end evaluation on the test (or validation) set.

    Steps:
      1. Run inference to collect y_true, y_pred, y_prob.
      2. Compute all metrics via compute_metrics().
      3. Print the full classification report.
      4. Save metrics to JSON (results/metrics/).
      5. Generate and save confusion matrix heatmap.
      6. Generate and save per-class ROC curves.

    Args:
        model       : Trained VGG16Classifier (must be loaded with best weights).
        test_loader : DataLoader for the held-out test set.
        device      : Compute device.
        class_names : Optional class name list (falls back to config).
        split_name  : 'test' or 'val' — used in filenames.

    Returns:
        Dictionary of all computed metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    logging.info(f"[Evaluate] Running inference on {split_name} set...")

    # ── 1. Collect predictions ────────────────────────────────────────────────
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    # ── 2. Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    # ── 3. Print classification report ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  CLASSIFICATION REPORT — {split_name.upper()} SET")
    print("=" * 70)
    print(metrics["classification_report"])
    print("=" * 70)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    : {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score  : {metrics['f1_weighted']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc_macro_ovr']:.4f}")
    print("=" * 70 + "\n")

    # ── 4. Save metrics JSON ──────────────────────────────────────────────────
    save_metrics(metrics, filename=f"{split_name}_metrics.json")

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        save=True,
        filename=f"{split_name}_confusion_matrix.png",
    )

    # ── 6. ROC curves ─────────────────────────────────────────────────────────
    plot_roc_curves(
        y_true, y_prob,
        class_names=class_names,
        save=True,
        filename=f"{split_name}_roc_curves.png",
    )

    logging.info(f"[Evaluate] All outputs saved for '{split_name}' split.")
    return metrics




# ==============================================================================
# main.py
# ==============================================================================
"""
main.py — Unified Entry Point
==============================
Orchestrates the complete pipeline:
  1. Parse CLI arguments
  2. Set up logging and seeds
  3. Build DataLoaders
  4. Build / load model
  5. Train (optional)
  6. Evaluate on validation and test sets
  7. Generate and save all plots

Usage Examples:
  # Full pipeline (train + eval):
  python main.py

  # Train only:
  python main.py --mode train

  # Evaluate only (load saved model):
  python main.py --mode eval

  # Fine-tune previously trained model:
  python main.py --mode train --fine_tune

  # Custom hyperparameters:
  python main.py --epochs 50 --lr 0.0001 --batch_size 16
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is on the Python path when running from any directory




# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    All arguments have sensible defaults drawn from py, so the project
    can be launched with just `python main.py`.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="VGG16 Skin Lesion Classification — MILK10k (ISIC Multimodal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Pipeline mode ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "full"],
        default="full",
        help="Pipeline mode: 'train' only | 'eval' only | 'full' (train + eval)",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Root directory of the dataset (one sub-folder per class)",
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,
                        help="Maximum training epochs")
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE,
                        help="Mini-batch size")
    parser.add_argument("--weight_decay",type=float, default=WEIGHT_DECAY,
                        help="Adam weight decay (L2 regularisation)")
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED,
                        help="Random seed for reproducibility")

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Unfreeze backbone for end-to-end fine-tuning",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights (train from scratch)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=BEST_MODEL_NAME,
        help="Filename of checkpoint to load for eval mode",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint specified by --checkpoint",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main orchestration function.

    Parses arguments, initialises the environment, and dispatches to the
    appropriate pipeline steps based on --mode.
    """
    args = parse_args()

    # ── Environment setup ─────────────────────────────────────────────────────
    ensure_dirs()
    setup_logging(LOG_FILE)
    set_seed(args.seed)

    device = get_device()

    # Apply CLI overrides to config (runtime patching)
    BATCH_SIZE    = args.batch_size
    NUM_EPOCHS    = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY  = args.weight_decay
    DATA_DIR      = Path(args.data_dir)

    freeze_features = not args.fine_tune
    pretrained      = not args.no_pretrained
    use_amp         = not args.no_amp

    logging.info("=" * 60)
    logging.info("  VGG16 Skin Lesion Classifier — MILK10k")
    logging.info("=" * 60)
    logging.info(f"  Mode         : {args.mode}")
    logging.info(f"  Device       : {device}")
    logging.info(f"  Epochs       : {args.epochs}")
    logging.info(f"  LR           : {args.lr}")
    logging.info(f"  Batch size   : {args.batch_size}")
    logging.info(f"  Fine-tune    : {args.fine_tune}")
    logging.info(f"  Pretrained   : {pretrained}")
    logging.info(f"  AMP          : {use_amp}")
    logging.info(f"  Seed         : {args.seed}")
    logging.info("=" * 60)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    logging.info("[Pipeline] Building DataLoaders...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=DATA_DIR
    )

    # ── Build Model ───────────────────────────────────────────────────────────
    logging.info("[Pipeline] Building model...")
    num_classes = len(class_names)
    model = build_model(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_features=freeze_features,
        device=device,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAIN MODE
    # ═══════════════════════════════════════════════════════════════════════════
    if args.mode in ("train", "full"):
        logging.info("[Pipeline] ── TRAINING PHASE ──")

        history = train_model(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            num_epochs   = args.epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            use_amp      = use_amp,
            resume_checkpoint = args.checkpoint if args.resume else None,
        )

        # Plot training curves
        plot_training_history(history, save=True)
        logging.info("[Pipeline] Training curves saved.")

        # ── Optional Phase-2 Fine-Tuning ──────────────────────────────────────
        if FINE_TUNE and freeze_features:
            logging.info("[Pipeline] ── FINE-TUNING PHASE (Phase 2) ──")
            model.unfreeze_backbone()

            history_ft = train_model(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = device,
                num_epochs   = max(args.epochs // 3, 5),   # Fewer epochs
                lr           = FINE_TUNE_LR,
                weight_decay = args.weight_decay,
                use_amp      = use_amp,
            )
            # Merge histories
            for key in history:
                history[key].extend(history_ft[key])

            plot_training_history(history, save=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # EVAL MODE — Load best checkpoint
    # ═══════════════════════════════════════════════════════════════════════════
    if args.mode in ("eval", "full"):
        logging.info("[Pipeline] ── EVALUATION PHASE ──")

        # Always evaluate using the best saved model
        logging.info(f"[Pipeline] Loading best checkpoint: {args.checkpoint}")
        try:
            model, _, _, _ = load_checkpoint(
                model,
                optimizer=None,
                filename=args.checkpoint,
                device=device,
            )
        except FileNotFoundError as e:
            logging.error(str(e))
            logging.error(
                "Run training first (python main.py --mode train) to generate a checkpoint."
            )
            sys.exit(1)

        # ── Validation set evaluation ─────────────────────────────────────────
        logging.info("[Pipeline] Evaluating on VALIDATION set...")
        evaluate_model(
            model       = model,
            test_loader = val_loader,
            device      = device,
            class_names = class_names,
            split_name  = "val",
        )

        # ── Test set evaluation ───────────────────────────────────────────────
        logging.info("[Pipeline] Evaluating on TEST set...")
        test_metrics = evaluate_model(
            model       = model,
            test_loader = test_loader,
            device      = device,
            class_names = class_names,
            split_name  = "test",
        )

        # Final summary printout
        print("\n" + "=" * 60)
        print("  FINAL TEST RESULTS")
        print("=" * 60)
        print(f"  Accuracy  : {test_metrics['accuracy']:.4f}  "
              f"({test_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision : {test_metrics['precision_weighted']:.4f}")
        print(f"  Recall    : {test_metrics['recall_weighted']:.4f}")
        print(f"  F1-Score  : {test_metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC   : {test_metrics['auc_roc_macro_ovr']:.4f}")
        print("=" * 60)
        print(f"\n  All outputs saved to: {RESULTS_DIR}")
        print("  Plots    ->", PLOTS_DIR)
        print("  Metrics  ->", METRICS_DIR)
        print("  Models   ->", MODELS_DIR)

    logging.info("[Pipeline] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()


