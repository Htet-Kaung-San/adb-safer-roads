"""
Worker script: run VLM on a specific chunk of segments.
Called by run_stage2_vlm_parallel.py — not meant to be run directly.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.vlm_inference import VLMRoadAnalyzer
from src.config import MAPILLARY_CACHE_DIR


def run(chunk_file, out_file, model_id, batch_size):
    seg_ids = json.load(open(chunk_file))
    print(f"Worker: {len(seg_ids)} segments, model={model_id}", flush=True)

    # Build image map
    segment_image_map = {}
    for seg_id in seg_ids:
        cache_dir = MAPILLARY_CACHE_DIR / seg_id
        imgs = list(cache_dir.glob("*.jpg")) if cache_dir.exists() else []
        if imgs:
            segment_image_map[seg_id] = imgs

    print(f"Worker: {len(segment_image_map)} segments have imagery", flush=True)

    analyzer = VLMRoadAnalyzer(model_id=model_id)
    results = analyzer.analyze_batch(segment_image_map, batch_size=batch_size)

    json.dump(results, open(out_file, "w"), indent=2)
    print(f"Worker done: {len(results)} results saved to {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run(args.chunk_file, args.out_file, args.model, args.batch)
