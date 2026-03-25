"""
Unified Zero-Shot Segmentation Pipeline.

Supports:
  - CLIPSeg
  - GroundingDINO + SAM

All outputs follow the shared format used across this repository:
  - results.csv
  - visuals/<dataset>/<image>.png
  - prediction_maps/<dataset>/<image>.{npy,png}
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

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


MODEL_CLIPSEG = "clipseg"
MODEL_GD_SAM = "groundingdino_sam"
MODEL_ALL = "all"
SUPPORTED_MODELS = [MODEL_CLIPSEG, MODEL_GD_SAM]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified zero-shot segmentation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=[*SUPPORTED_MODELS, MODEL_ALL],
        default=MODEL_ALL,
        help="Zero-shot model to run, or 'all' to run every model.",
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root", default="sample_data")
    parser.add_argument("--folder", default=None, help="Optional subfolder under data root.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="outputs/all_methods")
    parser.add_argument("--max-images", type=int, default=-1)

    # CLIPSeg options
    parser.add_argument(
        "--prompts",
        default="soybean leaf,soybean leaves,plant leaf,leaf,green leaf,green leaves",
        help="Comma-separated prompts used by CLIPSeg.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-otsu", action="store_true")

    # GroundingDINO + SAM options
    parser.add_argument(
        "--positive-prompts",
        default="soybean leaf,soybean leaves,leaf,soybean plant,plant leaf,foliage,green leaf",
    )
    parser.add_argument("--negative-prompts", default="grass,weed")
    parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--max-detections", type=int, default=200)
    parser.add_argument("--neg-iou-filter", type=float, default=0.5)
    parser.add_argument("--min-box-area-frac", type=float, default=2e-4)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--min-union-area-frac", type=float, default=0.001)
    parser.add_argument("--use-clip-filter", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-margin-thresh", type=float, default=0.0)
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam-checkpoint", default="sam_vit_b.pth")

    return parser.parse_args()


def otsu_threshold(prob_map: np.ndarray) -> float:
    hist, _ = np.histogram(prob_map.flatten(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = prob_map.size
    sum_b, w_b = 0.0, 0.0
    maximum, threshold = 0.0, 0.0
    sum1 = np.dot(hist, np.arange(256))
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum1 - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between >= maximum:
            threshold = i
            maximum = between
    return float(threshold) / 255.0


def run_clipseg(args: argparse.Namespace, dataset_dirs: List[Path], device: torch.device) -> Path:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    model_name = "CIDAS/clipseg-rd64-refined"
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    out_root = Path(args.output_root) / "zero_shot" / "CLIPSeg"
    records = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        pairs = discover_pairs(str(dataset_dir))
        if args.max_images > 0:
            pairs = pairs[: args.max_images]

        for img_path, gt_path in tqdm(pairs, desc=f"CLIPSeg ({dataset_name})"):
            stem = img_path.stem
            img = Image.open(img_path).convert("RGB")
            gt_mask = load_gt(gt_path)

            t0 = time.time()
            inputs = processor(
                text=prompts,
                images=[img] * len(prompts),
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(1)
                probs = torch.sigmoid(logits).max(dim=0).values
            probs = torch.nn.functional.interpolate(
                probs.unsqueeze(0).unsqueeze(0),
                size=(img.height, img.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()
            infer_time = time.time() - t0

            thr = otsu_threshold(probs) if args.use_otsu else args.threshold
            pred_mask = (probs >= thr).astype(np.uint8)

            if pred_mask.shape != gt_mask.shape:
                gt_img = Image.fromarray(gt_mask * 255)
                gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), resample=Image.NEAREST)
                gt_mask = (np.array(gt_img) > 127).astype(np.uint8)

            metrics = compute_metrics(pred_mask, gt_mask)
            save_prediction_map(pred_mask, out_root / "prediction_maps" / dataset_name / stem)
            save_visual(np.array(img), pred_mask, gt_mask, out_root / "visuals" / dataset_name / f"{stem}.png", "CLIPSeg")

            records.append({
                "dataset": dataset_name,
                "image": img_path.name,
                "prompts": "|".join(prompts),
                "threshold": thr,
                "inference_time_sec": infer_time,
                **metrics,
            })

    save_csv(records, out_root / "results.csv")
    return out_root


def maybe_download_sam_weights(model_type: str, ckpt_path: Path):
    if ckpt_path.exists():
        return
    url_map = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    url = url_map.get(model_type)
    if not url:
        raise FileNotFoundError("Unknown SAM model type for auto-download")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM weights: {url}")
    urllib.request.urlretrieve(url, ckpt_path)


def compose_grounding_caption(prompts: List[str]) -> str:
    parts = []
    for raw in prompts:
        candidate = raw.strip().rstrip(".")
        if candidate:
            parts.append(f"{candidate} .")
    return " ".join(parts)


def run_gdino_detection(processor, model, pil_img, prompts, box_thresh, text_thresh, max_det, device):
    caption = compose_grounding_caption(prompts)
    if not caption:
        return []

    inputs = processor(images=pil_img, text=[caption], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[pil_img.height, pil_img.width]], device=device)

    dets = []
    threshold_pairs = [(box_thresh, text_thresh)]
    fallback_box = max(min(box_thresh * 0.5, 0.15), 0.05)
    fallback_text = max(min(text_thresh * 0.5, 0.15), 0.05)
    if fallback_box < box_thresh or fallback_text < text_thresh:
        threshold_pairs.append((fallback_box, fallback_text))

    for cur_box, cur_text in threshold_pairs:
        try:
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.get("input_ids"),
                threshold=cur_box,
                text_threshold=cur_text,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=cur_box,
            )

        result = results[0] if len(results) > 0 else {}
        boxes = result.get("boxes")
        if boxes is None or len(boxes) == 0:
            continue

        scores_tensor = result.get("scores")
        labels = result.get("text_labels") or result.get("labels") or ["leaf"] * len(boxes)
        scores = (
            scores_tensor.detach().cpu().numpy()
            if isinstance(scores_tensor, torch.Tensor)
            else np.asarray(scores_tensor if scores_tensor is not None else [1.0] * len(boxes))
        )
        boxes_np = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)

        dets = []
        for idx in range(len(boxes_np)):
            label = labels[idx] if idx < len(labels) else ""
            dets.append((float(scores[idx]), label, boxes_np[idx]))
        dets.sort(key=lambda x: x[0], reverse=True)
        if dets:
            break

    return dets[:max_det]


def clip_box_to_image(box_xyxy: np.ndarray, hw: Tuple[int, int], pad_frac: float = 0.05) -> Tuple[int, int, int, int]:
    h, w = hw
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    if x1 <= x0:
        mid = 0.5 * (x0 + x1)
        x0, x1 = mid - 0.5, mid + 0.5
    if y1 <= y0:
        mid = 0.5 * (y0 + y1)
        y0, y1 = mid - 0.5, mid + 0.5
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x0 = max(0.0, x0 - pad_x)
    x1 = min(w - 1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(h - 1.0, y1 + pad_y)
    x0i, y0i = int(np.floor(x0)), int(np.floor(y0))
    x1i, y1i = int(np.ceil(x1)), int(np.ceil(y1))
    if x1i <= x0i:
        x1i = min(w - 1, x0i + 1)
    if y1i <= y0i:
        y1i = min(h - 1, y0i + 1)
    return x0i, y0i, x1i, y1i


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def refine_with_sam(predictor, image_np, boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return []
    predictor.set_image(image_np)
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32, device=predictor.device)
    transformed = predictor.transform.apply_boxes_torch(boxes_tensor, image_np.shape[:2])
    with torch.no_grad():
        masks_tensor, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
    return [(m.squeeze(0).cpu().numpy() > 0).astype(np.uint8) for m in masks_tensor]


def union_masks(masks):
    if not masks:
        return None
    union = np.zeros_like(masks[0])
    for m in masks:
        union = np.logical_or(union, m).astype(np.uint8)
    return union


def run_groundingdino_sam(args: argparse.Namespace, dataset_dirs: List[Path], device: torch.device) -> Path:
    try:
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
    except Exception as exc:
        raise ImportError("GroundingDINO not available. Install/upgrade transformers.") from exc

    try:
        from transformers import CLIPProcessor, CLIPModel
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as exc:
        raise ImportError("Missing GD-SAM dependencies. Install segment-anything and transformers extras.") from exc

    out_root = Path(args.output_root) / "zero_shot" / "GroundingDINO_SAM"

    pos_prompts = [p.strip() for p in args.positive_prompts.split(",") if p.strip()]
    neg_prompts = [p.strip() for p in args.negative_prompts.split(",") if p.strip()]

    processor = GroundingDinoProcessor.from_pretrained(args.gdino_model)
    gdino_model = GroundingDinoForObjectDetection.from_pretrained(args.gdino_model).to(device)
    gdino_model.eval()

    ckpt_path = Path(args.sam_checkpoint)
    maybe_download_sam_weights(args.sam_model_type, ckpt_path)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(ckpt_path))
    sam.to(device)
    predictor = SamPredictor(sam)

    clip_proc, clip_model = None, None
    if args.use_clip_filter:
        clip_proc = CLIPProcessor.from_pretrained(args.clip_model)
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
        clip_model.eval()

    records = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        pairs = discover_pairs(str(dataset_dir))
        if args.max_images > 0:
            pairs = pairs[: args.max_images]

        for img_path, gt_path in tqdm(pairs, desc=f"GD-SAM ({dataset_name})"):
            stem = img_path.stem
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)
            gt_mask = load_gt(gt_path)

            t0 = time.time()
            dets_pos = run_gdino_detection(
                processor,
                gdino_model,
                pil_img,
                pos_prompts,
                args.box_threshold,
                args.text_threshold,
                args.max_detections,
                device,
            )
            dets_neg = run_gdino_detection(
                processor,
                gdino_model,
                pil_img,
                neg_prompts,
                args.box_threshold,
                args.text_threshold,
                args.max_detections,
                device,
            ) if neg_prompts else []

            h, w = img_np.shape[:2]
            img_area = h * w
            filtered_pos = []
            for score, phrase, box in dets_pos:
                x0, y0, x1, y1 = box
                bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                if (bw * bh) / img_area < args.min_box_area_frac:
                    continue
                if max(bw / bh, bh / bw) > args.max_aspect_ratio:
                    continue
                filtered_pos.append((score, phrase, box))

            if dets_neg and filtered_pos:
                neg_boxes = [d[2] for d in dets_neg]
                kept = []
                for det in filtered_pos:
                    if not any(box_iou_xyxy(det[2], nb) >= args.neg_iou_filter for nb in neg_boxes):
                        kept.append(det)
                filtered_pos = kept

            if not filtered_pos and dets_pos:
                filtered_pos = dets_pos[: min(len(dets_pos), 6)]

            boxes_xyxy = [clip_box_to_image(d[2], img_np.shape[:2]) for d in filtered_pos]
            masks = refine_with_sam(predictor, img_np, boxes_xyxy) if boxes_xyxy else []

            if args.use_clip_filter and masks:
                with torch.no_grad():
                    pos_inputs = clip_proc(text=pos_prompts, return_tensors="pt", padding=True).to(device)
                    neg_inputs = clip_proc(
                        text=neg_prompts if neg_prompts else [""],
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    pos_feats = clip_model.get_text_features(**pos_inputs)
                    pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
                    neg_feats = clip_model.get_text_features(**neg_inputs)
                    neg_feats = neg_feats / (neg_feats.norm(dim=-1, keepdim=True) + 1e-6)

                kept = []
                for m in masks:
                    masked = img_np.copy()
                    masked[m == 0] = 127
                    inputs = clip_proc(images=Image.fromarray(masked), return_tensors="pt").to(device)
                    with torch.no_grad():
                        img_feats = clip_model.get_image_features(**inputs)
                        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                        pos_sim = (img_feats @ pos_feats.T).max().item()
                        neg_sim = (img_feats @ neg_feats.T).max().item() if neg_prompts else 0.0
                    if pos_sim - neg_sim >= args.clip_margin_thresh:
                        kept.append(m)
                if kept:
                    masks = kept

            pred_mask = union_masks(masks)
            if pred_mask is not None and (pred_mask.sum() / img_area) < args.min_union_area_frac:
                pred_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            if pred_mask is None:
                pred_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            infer_time = time.time() - t0

            if pred_mask.shape != gt_mask.shape:
                gt_img = Image.fromarray(gt_mask * 255)
                gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), resample=Image.NEAREST)
                gt_mask = (np.array(gt_img) > 127).astype(np.uint8)

            metrics = compute_metrics(pred_mask, gt_mask)
            save_prediction_map(pred_mask, out_root / "prediction_maps" / dataset_name / stem)
            save_visual(np.array(pil_img), pred_mask, gt_mask, out_root / "visuals" / dataset_name / f"{stem}.png", "GroundingDINO+SAM")

            records.append({
                "dataset": dataset_name,
                "image": img_path.name,
                "prompts": "|".join(pos_prompts),
                "inference_time_sec": infer_time,
                **metrics,
            })

    save_csv(records, out_root / "results.csv")
    return out_root


def main() -> None:
    args = parse_args()
    dataset_dirs = resolve_dataset_dirs(args.data_root, args.folder)
    device = torch.device(args.device)
    t0 = time.time()

    models_to_run = SUPPORTED_MODELS if args.model == MODEL_ALL else [args.model]
    out_roots = []
    for model_name in models_to_run:
        if model_name == MODEL_CLIPSEG:
            out_roots.append(run_clipseg(args, dataset_dirs, device))
        elif model_name == MODEL_GD_SAM:
            out_roots.append(run_groundingdino_sam(args, dataset_dirs, device))

    print(f"Done ({', '.join(models_to_run)}). Total time: {time.time() - t0:.2f}s")
    for out_root in out_roots:
        print(f"Results: {out_root}")


if __name__ == "__main__":
    main()
