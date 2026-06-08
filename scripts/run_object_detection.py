"""
YOLOv8 Object Detection on all cached Mapillary street images.

Runs on the GPU server (any single RTX A5000 is more than enough).
Counts pedestrians, cyclists, and motorcycles in each image to produce
objective, quantified VRU (Vulnerable Road User) density metrics.

COCO class IDs used:
    0 = person (pedestrian)
    1 = bicycle
    2 = car
    3 = motorcycle
    5 = bus
    7 = truck

VRU = person + bicycle + motorcycle
VRU ratio = VRU count / total detected objects

Outputs:
    outputs/scores/maharashtra_yolo.json
    outputs/scores/thailand_yolo.json

Each segment entry:
{
  "12345": {
    "pedestrian_count": 3,
    "cyclist_count": 1,
    "moto_count": 4,
    "car_count": 12,
    "total_count": 20,
    "vru_count": 8,
    "vru_ratio": 0.4,
    "images_processed": 2
  }
}

Usage (on GPU server, single GPU):
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_object_detection.py

Usage (all GPUs, data parallel):
    python3 scripts/run_object_detection.py --workers 6
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MAPILLARY_CACHE_DIR, SCORES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# COCO class IDs
PERSON_ID   = 0
BICYCLE_ID  = 1
CAR_ID      = 2
MOTO_ID     = 3
BUS_ID      = 5
TRUCK_ID    = 7

VRU_IDS     = {PERSON_ID, BICYCLE_ID, MOTO_ID}
VEHICLE_IDS = {CAR_ID, BUS_ID, TRUCK_ID}
ALL_IDS     = VRU_IDS | VEHICLE_IDS

CONFIDENCE_THRESHOLD = 0.30   # min YOLO confidence to count a detection


def _aggregate_detections(detections_per_image: list) -> dict:
    """Average counts across multiple images for a segment."""
    if not detections_per_image:
        return {
            "pedestrian_count": 0, "cyclist_count": 0, "moto_count": 0,
            "car_count": 0, "total_count": 0, "vru_count": 0,
            "vru_ratio": 0.0, "images_processed": 0,
        }

    totals = {
        "pedestrian_count": 0, "cyclist_count": 0, "moto_count": 0,
        "car_count": 0, "total_count": 0, "vru_count": 0,
    }
    for d in detections_per_image:
        for key in totals:
            totals[key] += d.get(key, 0)

    n = len(detections_per_image)
    avg = {k: round(v / n, 2) for k, v in totals.items()}
    avg["vru_ratio"] = round(avg["vru_count"] / max(avg["total_count"], 1), 4)
    avg["images_processed"] = n
    return avg


def _process_image(img_path: Path, model) -> dict:
    """Run YOLO on a single image, return count dict."""
    results = model(str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
    counts = {
        "pedestrian_count": 0, "cyclist_count": 0, "moto_count": 0,
        "car_count": 0, "total_count": 0, "vru_count": 0,
    }
    for result in results:
        for cls_id in result.boxes.cls.int().tolist():
            counts["total_count"] += 1
            if cls_id == PERSON_ID:
                counts["pedestrian_count"] += 1
                counts["vru_count"] += 1
            elif cls_id == BICYCLE_ID:
                counts["cyclist_count"] += 1
                counts["vru_count"] += 1
            elif cls_id == MOTO_ID:
                counts["moto_count"] += 1
                counts["vru_count"] += 1
            elif cls_id == CAR_ID:
                counts["car_count"] += 1
    return counts


def _get_region_seg_ids() -> dict:
    """Map segment IDs to regions using stage1 GeoJSON."""
    seg_to_region = {}
    for r in ["maharashtra", "thailand"]:
        for fname in [f"{r}_final.geojson", f"{r}_stage1.geojson"]:
            path = SCORES_DIR / fname
            if path.exists():
                import json as _json
                data = _json.load(open(path))
                for feat in data["features"]:
                    sid = str(feat["properties"].get("OBJECTID", ""))
                    if sid:
                        seg_to_region[sid] = r
                break
    return seg_to_region


def run(workers: int = 1, model_size: str = "l"):
    """
    Main detection loop.

    Args:
        workers: Number of parallel worker processes (each gets own GPU).
                 Set to 1 for single-GPU execution.
        model_size: YOLOv8 model size: n/s/m/l/x (l is a good balance)
    """
    from ultralytics import YOLO

    model_name = f"yolov8{model_size}.pt"
    logger.info(f"Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)

    # Collect all segment directories with cached images
    seg_dirs = sorted([
        d for d in MAPILLARY_CACHE_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.jpg"))
    ])
    logger.info(f"Found {len(seg_dirs)} segments with imagery")

    seg_to_region = _get_region_seg_ids()

    results_by_region = {"maharashtra": {}, "thailand": {}, "unknown": {}}
    total = len(seg_dirs)

    for i, seg_dir in enumerate(seg_dirs):
        seg_id = seg_dir.name
        img_paths = sorted(seg_dir.glob("*.jpg"))

        image_detections = []
        for img_path in img_paths[:3]:   # max 3 images per segment (consistent with VLM)
            try:
                det = _process_image(img_path, model)
                image_detections.append(det)
            except Exception as e:
                logger.warning(f"Detection failed for {img_path}: {e}")

        agg = _aggregate_detections(image_detections)
        region = seg_to_region.get(seg_id, "unknown")
        results_by_region[region][seg_id] = agg

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    # Save per-region JSON
    for region, data in results_by_region.items():
        if not data:
            continue
        out = SCORES_DIR / f"{region}_yolo.json"
        json.dump(data, open(out, "w"), indent=2)
        logger.info(f"Saved {region}: {len(data)} segments → {out}")

    # Summary statistics
    logger.info("\n=== YOLO Detection Summary ===")
    for region, data in results_by_region.items():
        if not data:
            continue
        vru_ratios = [d["vru_ratio"] for d in data.values()]
        ped_counts = [d["pedestrian_count"] for d in data.values()]
        high_vru = sum(1 for v in vru_ratios if v > 0.3)
        logger.info(f"{region.title()}:")
        logger.info(f"  Segments processed: {len(data):,}")
        logger.info(f"  Avg VRU ratio: {sum(vru_ratios)/max(len(vru_ratios),1):.3f}")
        logger.info(f"  High VRU (>30%): {high_vru} segments")
        logger.info(f"  Avg pedestrian count: {sum(ped_counts)/max(len(ped_counts),1):.1f}")

    logger.info("\nDone. Sync to Mac and re-run run_stage3_gnn.py or run_analysis.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes (1 per GPU)")
    parser.add_argument("--model", default="l",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (l=large, balanced speed/accuracy)")
    args = parser.parse_args()
    run(args.workers, args.model)
