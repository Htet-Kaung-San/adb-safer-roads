"""
Script: Train GNN and produce spatially-refined Speed Safety Scores.

Usage:
    python scripts/run_stage3_gnn.py --region all --epochs 150 --device cuda

Reads Stage-1 (and optionally Stage-2 VLM) outputs, builds the road network
graph, trains the GAT model, and writes the final scored GeoJSON.

Outputs:
    outputs/scores/maharashtra_final.geojson
    outputs/scores/thailand_final.geojson
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import geopandas as gpd
import pandas as pd

from src.gnn.graph_builder import build_graph, to_pyg_data
from src.gnn.model import build_and_train_gnn
from src.scoring.safe_system import compute_counterfactual_impact
from src.config import SCORES_DIR, SCORE_BANDS


def grade(score: float) -> tuple[str, str]:
    for lo, hi, g, lbl, _ in SCORE_BANDS:
        if lo <= score < hi:
            return g, lbl
    return "E", "Critical"


def run(region: str, epochs: int, device: str):
    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    for r in regions:
        print(f"\n{'='*60}")
        print(f"Stage 3 GNN: {r.title()}")
        print('='*60)

        stage1_path = SCORES_DIR / f"{r}_stage1.geojson"
        if not stage1_path.exists():
            print(f"Missing: {stage1_path}. Run run_stage1.py first.")
            continue

        gdf = gpd.read_file(stage1_path)

        # ── Merge VLM features if available ───────────────────────────────
        vlm_path = SCORES_DIR / f"{r}_stage2_vlm.json"
        has_vlm = vlm_path.exists()
        if has_vlm:
            print("Loading VLM features …")
            with open(vlm_path) as f:
                vlm_data = json.load(f)
            vlm_df = pd.DataFrame(vlm_data).T
            vlm_df.index = vlm_df.index.astype(str)
            gdf["_seg_id"] = gdf["OBJECTID"].astype(str)
            vlm_cols = [c for c in vlm_df.columns if c not in ("reasoning",)]
            gdf = gdf.merge(
                vlm_df[vlm_cols].add_suffix("_vlm") if False else vlm_df[vlm_cols],
                left_on="_seg_id", right_index=True, how="left"
            )
            gdf = gdf.drop(columns=["_seg_id"])
            print(f"  VLM features merged for {vlm_df.shape[0]} segments")
        else:
            print("No VLM features found — running GNN on tabular features only")

        # ── Build graph ────────────────────────────────────────────────────
        x, edge_index, y, node_index = build_graph(gdf, has_vlm=has_vlm)
        print(f"Node features: {x.shape[1]}")

        # ── Train GNN ─────────────────────────────────────────────────────
        model, refined_scores = build_and_train_gnn(
            gdf, x, edge_index, y, device=device, epochs=epochs
        )

        # ── Write final scores ─────────────────────────────────────────────
        gdf["gnn_speed_safety_score"] = refined_scores.numpy().round(2)
        gdf["final_score"] = (
            0.6 * gdf["speed_safety_score"] +   # Stage 1 tabular
            0.4 * gdf["gnn_speed_safety_score"]  # Stage 3 GNN
        ).round(2)

        # Add VLM weight if available
        vlm_features = [c for c in ["pedestrian_infra","cyclist_infra","roadside_activity",
                                      "road_condition","signage_quality","vru_exposure",
                                      "visibility_quality"] if c in gdf.columns]
        if vlm_features:
            # Adjust weighting to include VLM
            gdf["final_score"] = (
                0.45 * gdf["speed_safety_score"] +
                0.25 * gdf["gnn_speed_safety_score"] +
                0.30 * (gdf[vlm_features].fillna(0.5).mean(axis=1) * 100)
            ).round(2)

        grades_labels = [grade(s) for s in gdf["final_score"]]
        gdf["final_grade"] = [g for g, _ in grades_labels]
        gdf["final_label"] = [l for _, l in grades_labels]

        # Counterfactual impact on final score
        gdf = compute_counterfactual_impact(gdf, outcome="fatality")

        out_path = SCORES_DIR / f"{r}_final.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"\nFinal scores saved → {out_path}")

        # Summary
        print(f"\nFinal grade distribution:")
        print(gdf["final_grade"].value_counts().sort_index().to_string())

        critical = gdf[gdf["final_grade"] == "E"]
        print(f"\nCritical (E) segments: {len(critical)}")
        if len(critical):
            top_impact = critical.nlargest(5, "estimated_impact_index")
            print("Top 5 by impact index:")
            print(top_impact[["final_score","RoadClass","LandUse",
                               "SpeedLimit","F85thPercentileSpeed",
                               "nilsson_reduction_factor","estimated_impact_index"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all", choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.region, args.epochs, args.device)
