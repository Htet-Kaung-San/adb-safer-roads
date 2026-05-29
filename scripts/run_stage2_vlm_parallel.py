"""
Run VLM inference across multiple GPUs in parallel by splitting segments into chunks.

Usage:
    python3 scripts/run_stage2_vlm_parallel.py --gpus 0,3,4,5,6,7 --model Qwen/Qwen2-VL-7B-Instruct

Each GPU gets an equal slice of segments. Results saved per-GPU then merged.
Final output: outputs/scores/thailand_stage2_vlm.json + maharashtra_stage2_vlm.json
"""
import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MAPILLARY_CACHE_DIR, SCORES_DIR


def split_chunks(items, n):
    k, m = divmod(len(items), n)
    return [items[i*k + min(i,m):(i+1)*k + min(i+1,m)] for i in range(n)]


def run_parallel(gpus, model_id, batch_size):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all segment dirs with images
    seg_dirs = sorted([d for d in MAPILLARY_CACHE_DIR.iterdir()
                       if d.is_dir() and list(d.glob("*.jpg"))])
    print(f"Total segments with imagery: {len(seg_dirs)}")
    print(f"GPUs: {gpus}")
    print(f"Model: {model_id}\n")

    chunks = split_chunks(seg_dirs, len(gpus))

    # Write each chunk's segment IDs to a temp file
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = SCORES_DIR / f"_chunk_{i}.json"
        with open(chunk_path, "w") as f:
            json.dump([d.name for d in chunk], f)
        chunk_files.append(chunk_path)
        print(f"GPU {gpus[i]}: {len(chunk)} segments")

    print("\nLaunching parallel processes...\n")

    # Launch one subprocess per GPU
    procs = []
    log_files = []
    for i, (gpu, chunk_file) in enumerate(zip(gpus, chunk_files)):
        out_file = SCORES_DIR / f"_vlm_chunk_{i}.json"
        log_path = SCORES_DIR / f"_vlm_gpu{gpu}.log"
        log_files.append(log_path)
        cmd = [
            sys.executable, str(Path(__file__).parent / "_vlm_worker.py"),
            "--chunk-file", str(chunk_file),
            "--out-file", str(out_file),
            "--model", model_id,
            "--batch", str(batch_size),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        procs.append((p, gpu, out_file, log_path))
        print(f"Started GPU {gpu} (PID {p.pid}) → log: {log_path.name}")

    # Monitor progress
    print("\nMonitoring (Ctrl+C to stop watching, processes continue in background):")
    try:
        while any(p.poll() is None for p, *_ in procs):
            time.sleep(30)
            for p, gpu, out_file, log_path in procs:
                if out_file.exists():
                    done = len(json.load(open(out_file)))
                    status = "done" if p.poll() is not None else "running"
                    print(f"  GPU {gpu}: {done} segments [{status}]")
                else:
                    print(f"  GPU {gpu}: starting...")
    except KeyboardInterrupt:
        print("\nDetached from monitor. Processes still running.")
        print("Check logs with: tail -f outputs/scores/_vlm_gpu*.log")
        print("Merge results when done: python3 scripts/run_stage2_vlm_parallel.py --merge")
        return

    print("\nAll GPUs finished. Merging results...")
    merge_results(gpus, chunk_files)


def merge_results(gpus=None, chunk_files=None):
    """Merge all chunk outputs into final region files."""
    chunk_outputs = sorted(SCORES_DIR.glob("_vlm_chunk_*.json"))
    if not chunk_outputs:
        print("No chunk outputs found to merge.")
        return

    # Load stage1 scored GeoJSONs to map segment IDs to regions
    seg_to_region = {}
    for region in ["maharashtra", "thailand"]:
        stage1 = SCORES_DIR / f"{region}_stage1.geojson"
        if stage1.exists():
            import json as _json
            data = _json.load(open(stage1))
            for feat in data["features"]:
                seg_id = str(feat["properties"].get("OBJECTID", ""))
                if seg_id:
                    seg_to_region[seg_id] = region

    merged = {}
    for chunk_path in chunk_outputs:
        chunk_data = json.load(open(chunk_path))
        merged.update(chunk_data)

    print(f"Total segments with VLM features: {len(merged)}")

    # Split by region
    by_region = {"maharashtra": {}, "thailand": {}}
    unknown = {}
    for seg_id, features in merged.items():
        region = seg_to_region.get(seg_id)
        if region:
            by_region[region][seg_id] = features
        else:
            unknown[seg_id] = features

    for region, data in by_region.items():
        if data:
            out = SCORES_DIR / f"{region}_stage2_vlm.json"
            json.dump(data, open(out, "w"), indent=2)
            print(f"Saved {region}: {len(data)} segments → {out}")

    if unknown:
        out = SCORES_DIR / "unknown_stage2_vlm.json"
        json.dump(unknown, open(out, "w"), indent=2)
        print(f"Unknown region: {len(unknown)} segments → {out}")

    # Cleanup temp files
    for f in SCORES_DIR.glob("_chunk_*.json"):
        f.unlink()
    for f in SCORES_DIR.glob("_vlm_chunk_*.json"):
        f.unlink()

    print("\nDone! Now run: python3 scripts/run_stage3_gnn.py --region all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,3,4,5,6,7", help="Comma-separated GPU IDs to use")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--merge", action="store_true", help="Only merge existing chunk results")
    args = parser.parse_args()

    if args.merge:
        merge_results()
    else:
        gpus = [int(g) for g in args.gpus.split(",")]
        run_parallel(gpus, args.model, args.batch)
