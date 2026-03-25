"""
Classical Segmentation Baselines: Otsu, K-means, Watershed.

Usage:
    python classical_segmentation/classical_segmentation.py --data-root sample_data
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_utils import (
    discover_pairs,
    resolve_dataset_dirs,
    load_gt,
    compute_metrics,
    save_visual,
    save_csv,
    save_prediction_map,
)


def otsu(image_bgr):
    """Otsu threshold on green channel + morphological cleanup."""
    g = image_bgr[:, :, 1]
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return (mask > 127).astype(np.uint8)


def kmeans_seg(image_bgr):
    """K-means (k=2) on RGB; cluster with higher green = plant."""
    pixels = np.float32(image_bgr.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    plant = np.argmax(centers[:, 1])
    return (labels.reshape(image_bgr.shape[:2]) == plant).astype(np.uint8)


def watershed_seg(image_bgr):
    """Watershed with distance-transform markers."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(opening, k, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_bgr, markers)
    return (markers > 1).astype(np.uint8)


METHODS = {"Otsu": otsu, "KMeans": kmeans_seg, "Watershed": watershed_seg}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", "--data-root", dest="data_root", default="sample_data",
                        help="Dataset root. Accepts either a dataset folder (RGB/Ground Truth) "
                             "or a root containing per-crop folders.")
    parser.add_argument("--folder", default=None,
                        help="Optional subfolder to process under --data-root.")
    parser.add_argument("--output-dir", default="outputs/all_methods")
    args = parser.parse_args()

    dataset_dirs = resolve_dataset_dirs(args.data_root, args.folder)
    out = Path(args.output_dir) / "classical_segmentation"

    for method_name, method_fn in METHODS.items():
        records = []
        vis_dir = out / method_name / "visuals"

        for dataset_dir in dataset_dirs:
            pairs = discover_pairs(str(dataset_dir))
            dataset_name = dataset_dir.name

            for rgb_path, gt_path in pairs:
                rgb = np.array(Image.open(rgb_path).convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gt = load_gt(gt_path)

                pred = method_fn(bgr)
                metrics = compute_metrics(pred, gt)
                records.append({"dataset": dataset_name, "image": rgb_path.name, **metrics})
                save_visual(
                    rgb, pred, gt,
                    vis_dir / dataset_name / f"{rgb_path.stem}.png",
                    method_name,
                )
                save_prediction_map(
                    pred,
                    out / method_name / "prediction_maps" / dataset_name / rgb_path.stem,
                )

        save_csv(records, out / method_name / "results.csv")
        print(f"{method_name}: {len(records)} images processed -> {out / method_name}")


if __name__ == "__main__":
    main()
