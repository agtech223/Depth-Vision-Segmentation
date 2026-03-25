"""
Shared evaluation utilities for all segmentation methods.

Provides consistent metrics computation, visual output, and CSV logging.
Every method script imports from here to guarantee identical output format.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def discover_pairs(data_dir: str):
    """Return list of (rgb_path, gt_path) pairs from data_dir/RGB and data_dir/Ground Truth."""
    root = Path(data_dir)
    rgb_dir = root / "RGB"
    gt_dir = root / "Ground Truth"
    pairs = []
    for p in sorted(rgb_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            gt = gt_dir / f"{p.stem}_mask.npy"
            if gt.exists():
                pairs.append((p, gt))
    return pairs


def resolve_dataset_dirs(data_root: str | Path, folder: Optional[str] = None) -> List[Path]:
    """Resolve dataset directories containing RGB and Ground Truth subfolders."""
    root = Path(data_root)
    if folder:
        target = root / folder
        if not target.exists():
            raise FileNotFoundError(f"Folder not found: {target}")
        if not ((target / "RGB").exists() and (target / "Ground Truth").exists()):
            raise FileNotFoundError(
                f"Folder does not contain required subdirectories: {target}"
            )
        return [target]

    if (root / "RGB").exists() and (root / "Ground Truth").exists():
        return [root]

    candidates = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and (p / "RGB").exists() and (p / "Ground Truth").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset folders found under {root}. "
            "Expected either <data_root>/RGB + <data_root>/Ground Truth "
            "or subfolders containing those directories."
        )
    return candidates


def load_gt(gt_path) -> np.ndarray:
    """Load ground-truth mask as binary uint8 (0/1)."""
    mask = np.load(gt_path)
    return (mask > 0).astype(np.uint8)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute IoU, F1, Precision, Recall, Accuracy from binary masks."""
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * precision * recall / (precision + recall + eps),
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


# ------------------------------------------------------------------
# Visual output  (RGB | Prediction | Ground Truth)
# ------------------------------------------------------------------

def save_visual(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                save_path: Path, method_name: str = "") -> None:
    """Save a 3-panel figure: RGB | Prediction overlay | Ground Truth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=14)
    axes[0].axis("off")

    overlay = rgb.astype(np.float32).copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[pred == 1] = overlay[pred == 1] * 0.55 + green * 0.45
    axes[1].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[1].set_title(f"Prediction ({method_name})" if method_name else "Prediction", fontsize=14)
    axes[1].axis("off")

    gt_vis = np.zeros_like(rgb)
    gt_vis[gt == 1] = [0, 255, 0]
    axes[2].imshow(gt_vis)
    axes[2].set_title("Ground Truth", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------

def save_csv(records: List[Dict], save_path: Path) -> None:
    """Write per-image metrics to CSV."""
    if not records:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def save_prediction_map(pred: np.ndarray, base_path: Path) -> None:
    """Save predicted binary mask as both NPY and PNG using a shared naming scheme."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pred_uint8 = (pred > 0).astype(np.uint8)
    np.save(base_path.with_suffix(".npy"), pred_uint8)
    Image.fromarray((pred_uint8 * 255).astype(np.uint8)).save(base_path.with_suffix(".png"))
