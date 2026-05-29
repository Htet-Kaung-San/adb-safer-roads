"""
Script: Run Stage 1 (Safe System tabular scoring) for both regions.

Usage:
    python scripts/run_stage1.py [--region maharashtra|thailand|all]

Outputs:
    outputs/scores/maharashtra_stage1.geojson
    outputs/scores/thailand_stage1.geojson
    outputs/scores/combined_stage1.geojson
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.load_data import load_region, load_all
from src.scoring.safe_system import compute_speed_safety_score, compute_counterfactual_impact
from src.config import SCORES_DIR


def run(region: str):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    gdfs = []
    for r in regions:
        print(f"\n{'='*60}")
        print(f"Stage 1: {r.title()}")
        print('='*60)

        gdf = load_region(r, filter_valid=True)
        gdf = compute_speed_safety_score(gdf)
        gdf = compute_counterfactual_impact(gdf, outcome="fatality")

        out_path = SCORES_DIR / f"{r}_stage1.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"Saved → {out_path}")

        # Summary stats
        print(f"\nScore distribution ({r}):")
        print(gdf["score_grade"].value_counts().sort_index().to_string())
        print(f"\nTop 10 highest-risk segments:")
        top = gdf.nlargest(10, "speed_safety_score")[
            ["score_grade", "speed_safety_score", "RoadClass", "LandUse",
             "SpeedLimit", "F85thPercentileSpeed", "safe_system_threshold_kmh",
             "speed_excess_kmh", "RankedPercentile"]
        ]
        print(top.to_string(index=False))

        gdfs.append(gdf)

    if len(gdfs) > 1:
        import pandas as pd
        import geopandas as gpd
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
        out_path = SCORES_DIR / "combined_stage1.geojson"
        combined.to_file(out_path, driver="GeoJSON")
        print(f"\nCombined saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all", choices=["maharashtra", "thailand", "all"])
    args = parser.parse_args()
    run(args.region)
