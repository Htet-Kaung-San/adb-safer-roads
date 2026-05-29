"""
Quick smoke test: run VLM on 10 segments, print results.

Usage (server, 7B single GPU):
    python scripts/test_vlm.py --model Qwen/Qwen2-VL-7B-Instruct --n 10

Usage (server, 72B tensor parallel):
    python scripts/test_vlm.py --model Qwen/Qwen2-VL-72B-Instruct --tp 8 --vllm --n 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.vlm_inference import VLMRoadAnalyzer
from src.config import MAPILLARY_CACHE_DIR


def run(model_id: str, n: int, tensor_parallel: int, use_vllm: bool):
    cache_dirs = [d for d in MAPILLARY_CACHE_DIR.iterdir()
                  if d.is_dir() and list(d.glob("*.jpg"))][:n]

    if not cache_dirs:
        print(f"No cached images found in {MAPILLARY_CACHE_DIR}")
        sys.exit(1)

    print(f"Testing VLM on {len(cache_dirs)} segments using {model_id}\n")

    analyzer = VLMRoadAnalyzer(
        model_id=model_id,
        tensor_parallel=tensor_parallel,
        use_vllm=use_vllm,
    )

    segment_image_map = {d.name: list(d.glob("*.jpg")) for d in cache_dirs}
    results = analyzer.analyze_batch(segment_image_map, batch_size=n)

    print("\n=== Results ===")
    for seg_id, features in results.items():
        imgs = len(segment_image_map[seg_id])
        print(f"\nSegment {seg_id} ({imgs} images):")
        for k, v in features.items():
            if k != "reasoning":
                print(f"  {k:25s}: {v:.3f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        print(f"  {'reasoning':25s}: {features.get('reasoning', '')}")

    # Save test output
    out = Path("outputs/scores/vlm_test_output.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--n", type=int, default=10, help="Number of segments to test")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--vllm", action="store_true")
    args = parser.parse_args()
    run(args.model, args.n, args.tp, args.vllm)
