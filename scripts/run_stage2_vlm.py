"""
Script: Run VLM inference on cached Mapillary images (GPU cluster).

Usage (school server, 8× A5000):
    # For 7B model (single GPU, fast):
    python scripts/run_stage2_vlm.py --region all --model Qwen/Qwen2-VL-7B-Instruct

    # For 72B model (tensor parallel, all 8 GPUs):
    python scripts/run_stage2_vlm.py --region all --model Qwen/Qwen2-VL-72B-Instruct --tp 8 --vllm

Outputs:
    outputs/scores/maharashtra_stage2_vlm.json
    outputs/scores/thailand_stage2_vlm.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
from src.vision.vlm_inference import VLMRoadAnalyzer
from src.config import SCORES_DIR, MAPILLARY_CACHE_DIR


def run(region: str, model_id: str, tensor_parallel: int, use_vllm: bool, batch_size: int):
    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    analyzer = VLMRoadAnalyzer(
        model_id=model_id,
        tensor_parallel=tensor_parallel,
        use_vllm=use_vllm,
    )

    for r in regions:
        score_file = SCORES_DIR / f"{r}_stage1.geojson"
        if not score_file.exists():
            print(f"Missing: {score_file}. Run run_stage1.py first.")
            continue

        gdf = gpd.read_file(score_file)
        high_risk = gdf[gdf["score_grade"].isin(["D", "E"])].copy()

        # Build segment_id → image_paths map
        segment_image_map = {}
        for _, row in high_risk.iterrows():
            seg_id = str(row.get("OBJECTID", row.name))
            cache_dir = MAPILLARY_CACHE_DIR / seg_id
            if cache_dir.exists():
                paths = list(cache_dir.glob("*.jpg"))
                if paths:
                    segment_image_map[seg_id] = paths

        print(f"\n{r.title()}: {len(segment_image_map)} segments with imagery → VLM")

        results = analyzer.analyze_batch(segment_image_map, batch_size=batch_size)

        out_path = SCORES_DIR / f"{r}_stage2_vlm.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"VLM features saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all", choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (1=single GPU)")
    parser.add_argument("--vllm", action="store_true", help="Use vLLM engine (recommended for 72B)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per iteration")
    args = parser.parse_args()
    run(args.region, args.model, args.tp, args.vllm, args.batch)
