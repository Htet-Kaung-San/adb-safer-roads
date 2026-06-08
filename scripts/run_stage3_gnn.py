"""
Stage 3: Train GNN and produce spatially-refined Speed Safety Scores.

Usage (GPU server):
    python scripts/run_stage3_gnn.py --region all --epochs 300 --device cuda

Reads Stage-1 (and Stage-2 VLM) outputs, builds the road network graph,
trains the Graph Attention Network, computes Monte Carlo dropout uncertainty
estimates, and writes the final scored GeoJSON.

Outputs:
    outputs/scores/maharashtra_final.geojson
    outputs/scores/thailand_final.geojson

New in this version:
    - MC Dropout uncertainty: score_std, score_ci_low, score_ci_high, grade_uncertain
    - YOLO object detection features if available: yolo_vru_ratio, yolo_pedestrian_count
    - All fields needed by run_analysis.py (economic, archetypes, optimizer)
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
from src.config import SCORES_DIR, SCORE_BANDS, GNN_MC_SAMPLES


def _grade(score: float):
    for lo, hi, g, lbl, _ in SCORE_BANDS:
        if lo <= score < hi:
            return g, lbl
    return "E", "Critical"


def run(region: str, epochs: int, device: str, no_uncertainty: bool = False):
    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    for r in regions:
        print(f"\n{'='*65}")
        print(f"Stage 3 GNN: {r.title()}")
        print('='*65)

        stage1_path = SCORES_DIR / f"{r}_stage1.geojson"
        if not stage1_path.exists():
            print(f"Missing: {stage1_path}. Run run_stage1.py first.")
            continue

        gdf = gpd.read_file(stage1_path)

        # ── Merge VLM features ─────────────────────────────────────────────
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
                vlm_df[vlm_cols],
                left_on="_seg_id", right_index=True, how="left"
            )
            gdf = gdf.drop(columns=["_seg_id"])
            print(f"  VLM features merged for {vlm_df.shape[0]} segments "
                  f"({has_vlm and len(vlm_df)} / {len(gdf)} = "
                  f"{len(vlm_df)/len(gdf)*100:.1f}% coverage)")
        else:
            print("No VLM features found — running GNN on tabular features only")

        # ── Merge YOLO object detection features (if available) ────────────
        yolo_path = SCORES_DIR / f"{r}_yolo.json"
        has_yolo = yolo_path.exists()
        if has_yolo:
            print("Loading YOLO object detection features …")
            with open(yolo_path) as f:
                yolo_data = json.load(f)
            yolo_df = pd.DataFrame(yolo_data).T
            yolo_df.index = yolo_df.index.astype(str)
            gdf["_seg_id"] = gdf["OBJECTID"].astype(str)
            yolo_cols = [c for c in yolo_df.columns]
            gdf = gdf.merge(
                yolo_df[yolo_cols],
                left_on="_seg_id", right_index=True, how="left"
            )
            gdf = gdf.drop(columns=["_seg_id"])
            print(f"  YOLO features merged for {len(yolo_df)} segments")
        else:
            print("  No YOLO features — skipping object detection layer")

        # ── Build graph ────────────────────────────────────────────────────
        x, edge_index, y, node_index = build_graph(gdf, has_vlm=has_vlm)
        print(f"Node features: {x.shape[1]}")

        # ── Train GNN + compute uncertainty ───────────────────────────────
        compute_uncertainty = not no_uncertainty
        model, refined_scores, uncertainty = build_and_train_gnn(
            gdf, x, edge_index, y,
            device=device,
            epochs=epochs,
            compute_uncertainty=compute_uncertainty,
        )

        # ── Attach GNN scores ─────────────────────────────────────────────
        gdf["gnn_speed_safety_score"] = refined_scores.numpy().round(2)

        # Base final score (no VLM)
        gdf["final_score"] = (
            0.60 * gdf["speed_safety_score"] +
            0.40 * gdf["gnn_speed_safety_score"]
        ).round(2)

        # Adjust weights to include VLM if available
        vlm_features = [c for c in [
            "pedestrian_infra", "cyclist_infra", "roadside_activity",
            "road_condition", "signage_quality", "vru_exposure", "visibility_quality"
        ] if c in gdf.columns]

        if vlm_features:
            vlm_mean = gdf[vlm_features].fillna(0.5).mean(axis=1) * 100
            gdf["vlm_mean_score"] = vlm_mean.round(2)
            gdf["final_score"] = (
                0.45 * gdf["speed_safety_score"] +
                0.25 * gdf["gnn_speed_safety_score"] +
                0.30 * vlm_mean
            ).clip(0, 100).round(2)
        else:
            gdf["vlm_mean_score"] = np.nan

        # ── Uncertainty fields ─────────────────────────────────────────────
        if uncertainty:
            gdf["score_std"]       = uncertainty["score_std"].numpy().round(2)
            gdf["score_ci_low"]    = uncertainty["score_ci_low"].numpy().round(2)
            gdf["score_ci_high"]   = uncertainty["score_ci_high"].numpy().round(2)
            gdf["grade_uncertain"] = uncertainty["grade_uncertain"].numpy()

            # Combine CI with final score adjustment
            # Shift CI to be centred on final_score rather than GNN score
            ci_half = (gdf["score_ci_high"] - gdf["score_ci_low"]) / 2
            gdf["score_ci_low"]  = (gdf["final_score"] - ci_half).clip(0, 100).round(2)
            gdf["score_ci_high"] = (gdf["final_score"] + ci_half).clip(0, 100).round(2)
        else:
            gdf["score_std"]       = np.nan
            gdf["score_ci_low"]    = np.nan
            gdf["score_ci_high"]   = np.nan
            gdf["grade_uncertain"] = False

        # ── Grades ────────────────────────────────────────────────────────
        grades_labels = [_grade(s) for s in gdf["final_score"]]
        gdf["final_grade"] = [g for g, _ in grades_labels]
        gdf["final_label"] = [l for _, l in grades_labels]

        # ── Nilsson counterfactual ─────────────────────────────────────────
        gdf = compute_counterfactual_impact(gdf, outcome="fatality")

        # ── Save ──────────────────────────────────────────────────────────
        out_path = SCORES_DIR / f"{r}_final.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"\nFinal scores saved → {out_path}")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\nFinal grade distribution:")
        print(gdf["final_grade"].value_counts().sort_index().to_string())

        grade_d = gdf[gdf["final_grade"] == "D"]
        grade_e = gdf[gdf["final_grade"] == "E"]
        print(f"\nGrade D (Unsafe):   {len(grade_d):,}")
        print(f"Grade E (Critical): {len(grade_e):,}")

        if len(grade_e):
            top = grade_e.nlargest(3, "estimated_impact_index")
            print("\nTop Critical (E) segments:")
            print(top[[
                "final_score", "RoadClass", "LandUse",
                "SpeedLimit", "F85thPercentileSpeed",
                "nilsson_reduction_factor", "estimated_impact_index"
            ]].to_string())

        if uncertainty:
            n_uncertain = gdf["grade_uncertain"].sum()
            pct = n_uncertain / len(gdf) * 100
            print(f"\nGrade-uncertain segments (95% CI crosses grade boundary): "
                  f"{n_uncertain} ({pct:.1f}%)")
            print(f"Mean score std dev: {gdf['score_std'].dropna().mean():.2f} points")

        print(f"\nNext: rsync this to Mac, then run:")
        print(f"  python scripts/run_analysis.py --region {r}")
        print(f"  python scripts/generate_maps.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all",
                        choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-uncertainty", action="store_true",
                        help="Skip MC dropout uncertainty estimation (faster)")
    args = parser.parse_args()
    run(args.region, args.epochs, args.device, args.no_uncertainty)
