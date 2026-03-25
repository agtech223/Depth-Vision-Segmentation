"""
Unified Depth Inference and Crop Segmentation Pipeline.

Supports three monocular depth estimation models:
  - Depth Anything V2 Large  (depth-anything/Depth-Anything-V2-Large-hf)
  - Depth Pro               (apple/DepthPro-hf)
  - DPT-Large               (Intel/dpt-large)

Each model runs depth inference followed by MiniBatchKMeans clustering
(batch_size=4096) to separate foreground plants from background soil.
For each image a single PNG is saved showing a green overlay of the
plant cluster on the original RGB image.

Reference
---------
Afzaal et al., "Zero-Shot Crop Segmentation via 3D Depth-Aware Vision
Pipeline using Unsupervised Clustering", 2025.

Usage
-----
    python pipeline.py --model depth_anything
    python pipeline.py --model depth_pro
    python pipeline.py --model dpt_large

    # Single crop folder, test mode (5 images):
    python pipeline.py --model depth_anything --folder Kidneybeans_Final --test

Directory layout expected under --data-root (default: sample_data):
    <data_root>/
        <crop_folder>/
            RGB/          <- JPEG / PNG images
            Ground Truth/ <- optional ground-truth masks
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_utils import load_gt, compute_metrics, save_visual, save_csv, save_prediction_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DEPTH_ANYTHING = "depth_anything"
MODEL_DEPTH_PRO = "depth_pro"
MODEL_DPT_LARGE = "dpt_large"
MODEL_ALL = "all"

SUPPORTED_MODELS = [MODEL_DEPTH_ANYTHING, MODEL_DEPTH_PRO, MODEL_DPT_LARGE]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Runtime configuration shared across all pipeline components."""

    model: str = MODEL_DEPTH_ANYTHING
    data_root: Path = Path("sample_data")
    output_root: Path = Path("")
    rgb_subdir: str = "RGB"

    test_mode: bool = False
    test_image_count: int = 5

    max_cluster_samples: int = 80_000
    kmeans_batch_size: int = 4_096
    kmeans_max_iter: int = 40
    random_seed: Optional[int] = 42
    overlay_opacity: float = 0.45
    precision: str = "auto"

    device: torch.device = field(default_factory=lambda: _default_device())

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        if not self.output_root or str(self.output_root) == ".":
            self.output_root = Path("outputs") / "all_methods" / "depth_models" / self.model
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, relative: Path | str) -> Path:
        path = self.output_root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path


def _default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        from transformers import infer_device
        return torch.device(infer_device())
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference Modules
# ---------------------------------------------------------------------------

class BaseInferenceModule(ABC):
    """Abstract base class for depth inference modules."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = config.device
        self.has_cuda = self.device.type == "cuda"
        if self.has_cuda:
            torch.backends.cudnn.benchmark = True
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights and processor."""

    @abstractmethod
    def infer(self, image_path: Path) -> Tuple[np.ndarray, Image.Image, float, Dict]:
        """Run depth inference on a single image.

        Returns
        -------
        depth : np.ndarray (H, W), float32
            Depth map where lower values = closer objects.
        image : PIL.Image.Image
        elapsed : float
            Forward-pass time in seconds.
        metadata : dict
            Model-specific metadata (e.g. FOV for Depth Pro).
        """

    def _use_float16(self) -> bool:
        return self.has_cuda and self.config.precision != "float32"

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self._use_float16() else torch.float32

    def _timed_forward(self, model, inputs) -> Tuple[object, float]:
        if self.has_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self._use_float16():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        if self.has_cuda:
            torch.cuda.synchronize()
        return outputs, time.perf_counter() - t0


class DepthAnythingInferenceModule(BaseInferenceModule):
    """Depth Anything V2 Large — relative inverse depth (disparity).

    Output is inverted so that lower values = closer objects.
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f"[INFO] Loading Depth Anything V2 Large ({self.MODEL_ID}) ...")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.MODEL_ID, torch_dtype=self._dtype()
        ).to(self.device)
        self.model.eval()
        print(f"[INFO] Model ready on {self.device}")

    def infer(self, image_path: Path) -> Tuple[np.ndarray, Image.Image, float, Dict]:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs, elapsed = self._timed_forward(self.model, inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic", align_corners=False,
        )
        depth_raw = prediction.squeeze().cpu().numpy().astype(np.float32)
        max_val = float(np.max(depth_raw))
        depth = (max_val - depth_raw) if max_val > 0 else depth_raw
        return depth, image, elapsed, {}


class DepthProInferenceModule(BaseInferenceModule):
    """Apple Depth Pro — metric depth in metres."""

    MODEL_ID = "apple/DepthPro-hf"

    def _load_model(self) -> None:
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        print(f"[INFO] Loading Depth Pro ({self.MODEL_ID}) ...")
        self.processor = DepthProImageProcessorFast.from_pretrained(self.MODEL_ID)
        self.model = DepthProForDepthEstimation.from_pretrained(
            self.MODEL_ID, torch_dtype=self._dtype()
        ).to(self.device)
        self.model.eval()
        print(f"[INFO] Model ready on {self.device}")

    def infer(self, image_path: Path) -> Tuple[np.ndarray, Image.Image, float, Dict]:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs, elapsed = self._timed_forward(self.model, inputs)
        processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        depth = processed["predicted_depth"].cpu().numpy().astype(np.float32)
        metadata = {
            "fov": float(processed["field_of_view"]),
            "focal_length": float(processed["focal_length"]),
        }
        return depth, image, elapsed, metadata


class DPTLargeInferenceModule(BaseInferenceModule):
    """Intel DPT-Large (MiDaS 3.0) — relative inverse depth.

    Output is inverted so that lower values = closer objects.
    """

    MODEL_ID = "Intel/dpt-large"

    def _load_model(self) -> None:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        print(f"[INFO] Loading DPT-Large ({self.MODEL_ID}) ...")
        self.processor = DPTImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = DPTForDepthEstimation.from_pretrained(
            self.MODEL_ID, torch_dtype=self._dtype()
        ).to(self.device)
        self.model.eval()
        print(f"[INFO] Model ready on {self.device}")

    def infer(self, image_path: Path) -> Tuple[np.ndarray, Image.Image, float, Dict]:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs, elapsed = self._timed_forward(self.model, inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic", align_corners=False,
        )
        depth_raw = prediction.squeeze().cpu().numpy().astype(np.float32)
        max_val = float(np.max(depth_raw))
        depth = (max_val - depth_raw) if max_val > 0 else depth_raw
        return depth, image, elapsed, {}


def build_inference_module(config: PipelineConfig) -> BaseInferenceModule:
    """Factory: return the correct inference module for the chosen model."""
    modules = {
        MODEL_DEPTH_ANYTHING: DepthAnythingInferenceModule,
        MODEL_DEPTH_PRO: DepthProInferenceModule,
        MODEL_DPT_LARGE: DPTLargeInferenceModule,
    }
    if config.model not in modules:
        raise ValueError(f"Unknown model '{config.model}'. Choose from: {SUPPORTED_MODELS}")
    return modules[config.model](config)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class ClusteringModule:
    """Two-cluster MiniBatchKMeans segmentation on a depth map.

    Cluster 0 -> plants (closer, lower depth)
    Cluster 1 -> soil   (further, higher depth)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def cluster(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster a depth map into two classes.

        Returns (cluster_map [H,W], median_centers [2,]).
        """
        flat = depth.reshape(-1).astype(np.float32)

        if float(np.std(flat)) < 1e-6:
            return (
                np.zeros_like(depth, dtype=np.uint8),
                np.array([flat[0], flat[0]], dtype=np.float32),
            )

        sample_count = min(flat.size, self.config.max_cluster_samples)
        indices = (
            np.arange(flat.size, dtype=np.int64)
            if sample_count == flat.size
            else self.rng.choice(flat.size, size=sample_count, replace=False)
        )
        samples = flat[indices].reshape(-1, 1)

        kmeans = MiniBatchKMeans(
            n_clusters=2,
            batch_size=min(self.config.kmeans_batch_size, sample_count),
            max_iter=self.config.kmeans_max_iter,
            n_init=5,
            init="k-means++",
            random_state=self.config.random_seed,
        )
        kmeans.fit(samples)

        centers_sorted = np.sort(kmeans.cluster_centers_.reshape(-1))
        distances = np.abs(flat[:, None] - centers_sorted[None, :])
        assignments = np.argmin(distances, axis=1).astype(np.uint8).reshape(depth.shape)

        centers = np.array(
            [
                float(np.median(flat[assignments.reshape(-1) == 0]))
                if np.any(assignments == 0) else float(np.median(flat)),
                float(np.median(flat[assignments.reshape(-1) == 1]))
                if np.any(assignments == 1) else float(np.median(flat)),
            ],
            dtype=np.float32,
        )
        return assignments, centers


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------

class DepthOutputWriter:
    """Saves plant-cluster overlays on the original RGB image as PNG."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def save_plant_overlay(
        self, clusters: np.ndarray, base_name: str,
        output_dir: Path, image: Image.Image,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plant_mask = (clusters == 0).astype(bool)
        img_arr = np.array(image, dtype=np.float32)
        if img_arr.ndim == 2:
            img_arr = np.repeat(img_arr[..., None], 3, axis=2)
        overlay = img_arr.copy()
        green = np.array([0, 255, 0], dtype=np.float32)
        overlay[plant_mask] = (
            overlay[plant_mask] * (1.0 - self.config.overlay_opacity)
            + green * self.config.overlay_opacity
        )
        result = np.clip(overlay, 0, 255).astype(np.uint8)
        Image.fromarray(result).save(output_dir / f"{base_name}_plant_overlay.png")


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Coordinates data loading, inference, clustering, and output."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.inference = build_inference_module(config)
        self.clusterer = ClusteringModule(config)
        self.writer = DepthOutputWriter(config)
        self.folder_logs: Dict[str, List[Dict]] = {}

    def _discover_folders(self) -> List[Path]:
        folders = []

        # Support direct dataset layout: <data_root>/RGB + <data_root>/Ground Truth
        direct_rgb = self.config.data_root / self.config.rgb_subdir
        if direct_rgb.is_dir():
            folders.append(direct_rgb)

        # Support multi-folder layout: <data_root>/<crop>/RGB
        for crop_dir in sorted(self.config.data_root.iterdir()):
            if not crop_dir.is_dir():
                continue
            rgb_dir = crop_dir / self.config.rgb_subdir
            if rgb_dir.is_dir():
                folders.append(rgb_dir)
        return folders

    def _list_images(self, folder: Path) -> List[Path]:
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if self.config.test_mode:
            images = images[: self.config.test_image_count]
        return images

    def process_folder(self, rgb_dir: Path) -> None:
        images = self._list_images(rgb_dir)
        if not images:
            print(f"[WARNING] No images found in {rgb_dir}")
            return

        crop_name = rgb_dir.parent.name
        gt_dir = rgb_dir.parent / "Ground Truth"
        records: List[Dict] = []

        for image_path in tqdm(images, desc=crop_name, unit="img"):
            try:
                depth, image, elapsed, metadata = self.inference.infer(image_path)
                base = image_path.stem
                clusters, centers = self.clusterer.cluster(depth)
                pred = (clusters == 0).astype(np.uint8)

                # Persist raw depth maps for downstream clustering comparisons.
                depth_dir = self.config.output_root / "depth_maps" / crop_name
                depth_dir.mkdir(parents=True, exist_ok=True)
                np.save(depth_dir / f"{base}.npy", depth.astype(np.float32))

                gt_path = gt_dir / f"{base}_mask.npy"
                if not gt_path.exists():
                    print(f"[WARNING] Missing GT for {image_path.name}; skipping")
                    continue

                gt = load_gt(gt_path)
                rgb_np = np.array(image)

                save_prediction_map(
                    pred,
                    self.config.output_root / "prediction_maps" / crop_name / base,
                )
                save_visual(
                    rgb_np,
                    pred,
                    gt,
                    self.config.output_root / "visuals" / crop_name / f"{base}.png",
                    self.config.model,
                )

                metrics = compute_metrics(pred, gt)
                record: Dict = {
                    "dataset": crop_name,
                    "image": image_path.name,
                    "inference_time_s": round(elapsed, 6),
                    "plant_pixels": int(np.sum(clusters == 0)),
                    "soil_pixels": int(np.sum(clusters == 1)),
                    "plant_depth_center": round(float(centers[0]), 6),
                }
                record.update(metrics)
                record.update(metadata)
                records.append(record)
            except Exception as exc:
                print(f"[ERROR] {image_path.name}: {exc}")

        if records:
            csv_path = self.config.output_root / "results.csv"
            existing = self.folder_logs.get("__all__", [])
            existing.extend(records)
            self.folder_logs["__all__"] = existing
            save_csv(existing, csv_path)
            self.folder_logs[crop_name] = records

    def run(self, target_folder: Optional[str] = None) -> None:
        _print_header("DEPTH INFERENCE AND SEGMENTATION PIPELINE")
        print(f"  Model  : {self.config.model}")
        print(f"  Device : {self.config.device}")
        print(f"  Output : {self.config.output_root}")
        if self.config.test_mode:
            print(f"  [TEST MODE] {self.config.test_image_count} images per folder")

        folders = self._discover_folders()
        if target_folder:
            folders = [f for f in folders if f.parent.name == target_folder]
        if not folders:
            print(f"[WARNING] No RGB directories found under {self.config.data_root}")
            return

        print("\nFolders queued:")
        for f in folders:
            print(f"  - {f.parent.name}")
        print()

        for folder in tqdm(folders, desc="Folders", unit="folder"):
            self.process_folder(folder)

        self._print_summary()

    def _print_summary(self) -> None:
        _print_header("PIPELINE SUMMARY")
        all_records = self.folder_logs.get("__all__", [])
        total_images = len(all_records)
        total_time = float(sum(r["inference_time_s"] for r in all_records))

        for crop, records in self.folder_logs.items():
            if crop == "__all__":
                continue
            times = [r["inference_time_s"] for r in records]
            print(f"\n{crop}:")
            print(f"  Images processed : {len(times)}")
            print(f"  Mean time/image  : {np.mean(times):.4f}s")
            print(f"  Total time       : {sum(times):.2f}s")
        if total_images:
            print(f"\nTotal images : {total_images}")
            print(f"Total time   : {total_time:.2f}s")
            print(f"Avg/image    : {total_time / total_images:.4f}s")
        _print_header()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _write_csv(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _print_header(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified depth inference and crop segmentation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=[*SUPPORTED_MODELS, MODEL_ALL],
                        default=MODEL_ALL, help="Depth model to use, or 'all' to run every model.")
    parser.add_argument("--data-root", type=str, default="sample_data",
                        help="Root directory containing per-crop folders.")
    parser.add_argument("--output-root", type=str, default="outputs/all_methods",
                        help="Shared output root for all methods.")
    parser.add_argument("--rgb-subdir", type=str, default="RGB")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process only this crop folder.")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only first --test-count images.")
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--precision", choices=["auto", "float32"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_to_run = SUPPORTED_MODELS if args.model == MODEL_ALL else [args.model]

    for model_name in models_to_run:
        config = PipelineConfig(
            model=model_name,
            data_root=Path(args.data_root),
            output_root=Path(args.output_root) / "depth_models" / model_name,
            rgb_subdir=args.rgb_subdir,
            test_mode=args.test,
            test_image_count=args.test_count,
            precision=args.precision,
        )
        orchestrator = PipelineOrchestrator(config)
        orchestrator.run(target_folder=args.folder)


if __name__ == "__main__":
    main()
