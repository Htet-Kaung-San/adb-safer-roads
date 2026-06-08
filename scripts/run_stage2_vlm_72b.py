"""
Run Qwen2-VL-72B inference using vLLM with tensor parallelism across multiple GPUs.

72B model requires tensor parallelism — all GPUs work as ONE big GPU.
This is different from the 7B parallel script which uses data parallelism.

Usage (6 free GPUs — 0,3,4,5,6,7):
    CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 python3 scripts/run_stage2_vlm_72b.py

VRAM requirement: 72B × fp16 = ~144GB → needs at least 6 × 24GB GPUs

Outputs:
    outputs/scores/maharashtra_stage2_vlm.json  (overwrites 7B results)
    outputs/scores/thailand_stage2_vlm.json
"""
import json
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MAPILLARY_CACHE_DIR, SCORES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"


def get_segment_image_map():
    """Load all cached segment images."""
    seg_dirs = [d for d in MAPILLARY_CACHE_DIR.iterdir()
                if d.is_dir() and list(d.glob("*.jpg"))]
    seg_map = {d.name: list(d.glob("*.jpg")) for d in seg_dirs}
    logger.info(f"Found {len(seg_map)} segments with imagery")
    return seg_map


def split_by_region(seg_map: dict) -> dict:
    """Map segment IDs to regions using stage1 GeoJSON files."""
    seg_to_region = {}
    for region in ["maharashtra", "thailand"]:
        for fname in [f"{region}_final.geojson", f"{region}_stage1.geojson"]:
            path = SCORES_DIR / fname
            if path.exists():
                data = json.load(open(path))
                for feat in data["features"]:
                    sid = str(feat["properties"].get("OBJECTID", ""))
                    if sid:
                        seg_to_region[sid] = region
                break
    return seg_to_region


def run():
    # Count visible GPUs
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_gpus = len(visible.split(",")) if visible else 8
    logger.info(f"Tensor parallel size: {n_gpus} GPUs")
    logger.info(f"VRAM available: ~{n_gpus * 24}GB | Required: ~144GB")

    if n_gpus < 6:
        logger.error(f"Need at least 6 GPUs for 72B model. Got {n_gpus}. "
                     f"Set CUDA_VISIBLE_DEVICES=0,3,4,5,6,7")
        sys.exit(1)

    seg_map = get_segment_image_map()
    seg_to_region = split_by_region(seg_map)

    logger.info(f"Loading {MODEL_ID} via HuggingFace device_map=auto — this takes ~5-10 minutes...")

    from src.vision.vlm_inference import VLMRoadAnalyzer
    analyzer = VLMRoadAnalyzer(
        model_id=MODEL_ID,
        device="auto",   # HuggingFace distributes layers across all visible GPUs
        use_vllm=False,
    )

    logger.info("Model loaded. Starting inference...")
    results = analyzer.analyze_batch(seg_map, batch_size=32)

    # Split by region and save
    by_region = {"maharashtra": {}, "thailand": {}}
    unknown = {}
    for seg_id, features in results.items():
        region = seg_to_region.get(seg_id)
        if region in by_region:
            by_region[region][seg_id] = features
        else:
            unknown[seg_id] = features

    for region, data in by_region.items():
        if data:
            out = SCORES_DIR / f"{region}_stage2_vlm.json"
            json.dump(data, open(out, "w"), indent=2)
            logger.info(f"Saved {region}: {len(data)} segments → {out}")

    if unknown:
        out = SCORES_DIR / "unknown_stage2_vlm.json"
        json.dump(unknown, open(out, "w"), indent=2)
        logger.info(f"Unknown region: {len(unknown)} segments")

    total = sum(len(v) for v in by_region.values())
    logger.info(f"\nDone! {total} segments processed with 72B model.")
    logger.info("Now run: python3 scripts/run_stage3_gnn.py --region all --epochs 300")


if __name__ == "__main__":
    run()
