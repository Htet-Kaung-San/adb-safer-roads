"""
Script: Fetch Mapillary images for scored segments.

Usage:
    export MAPILLARY_TOKEN="MLY|..."
    python scripts/run_stage2_fetch.py --region all --threads 32 --grades C,D,E

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


def run(region: str, threads: int, grades: list):
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
        # Use final scores if available, else stage1
        score_file = SCORES_DIR / f"{r}_final.geojson"
        grade_col = "final_grade"
        if not score_file.exists():
            score_file = SCORES_DIR / f"{r}_stage1.geojson"
            grade_col = "score_grade"
        if not score_file.exists():
            print(f"No scored output found for {r}. Run run_stage1.py first.")
            continue

        print(f"\nFetching Mapillary images for {r.title()} (grades: {','.join(grades)}) …")
        gdf = gpd.read_file(score_file)

        target = gdf[gdf[grade_col].isin(grades)].copy()

        # Skip segments already cached
        already = sum(
            1 for _, row in target.iterrows()
            if list((MAPILLARY_CACHE_DIR / str(row.get("OBJECTID", ""))).glob("*.jpg"))
        )
        print(f"  {len(target):,} segments targeted | {already:,} already cached | "
              f"{len(target)-already:,} to fetch")

        results = batch_fetch_images(target, token=token, num_threads=threads)
        with_images = sum(1 for p in results.values() if p)
        print(f"  Done: {with_images}/{len(results)} segments have imagery")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all", choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--grades", default="D,E",
                        help="Comma-separated grades to fetch e.g. C,D,E (default: D,E)")
    args = parser.parse_args()
    run(args.region, args.threads, [g.strip() for g in args.grades.split(",")])
