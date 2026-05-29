"""
Script: Fetch Mapillary images for all scored segments.

Usage:
    export MAPILLARY_TOKEN="MLY|..."
    python scripts/run_stage2_fetch.py --region all --threads 32

Reads Stage-1 outputs and downloads imagery into:
    data/processed/mapillary_cache/<segment_id>/<image_id>.jpg
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
from src.vision.mapillary import batch_fetch_images
from src.config import SCORES_DIR, MAPILLARY_CACHE_DIR


def run(region: str, threads: int):
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        print("ERROR: Set MAPILLARY_TOKEN environment variable first.")
        sys.exit(1)

    MAPILLARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    for r in regions:
        score_file = SCORES_DIR / f"{r}_stage1.geojson"
        if not score_file.exists():
            print(f"Stage-1 output not found: {score_file}. Run run_stage1.py first.")
            continue

        print(f"\nFetching Mapillary images for {r.title()} …")
        gdf = gpd.read_file(score_file)

        # Only fetch for D/E (Critical/Unsafe) segments to save API quota
        high_risk = gdf[gdf["score_grade"].isin(["D", "E"])].copy()
        print(f"  {len(high_risk):,} high-risk segments to fetch imagery for")

        results = batch_fetch_images(
            high_risk,
            token=token,
            num_threads=threads,
        )

        with_images = sum(1 for p in results.values() if p)
        print(f"  Done: {with_images}/{len(results)} segments have imagery")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all", choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    run(args.region, args.threads)
