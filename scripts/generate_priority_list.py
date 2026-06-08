"""
Generate ranked intervention priority lists as CSV files.

Produces:
  outputs/priority/priority_list_all.csv         — all risk segments, globally ranked
  outputs/priority/priority_list_thailand.csv    — Thailand only
  outputs/priority/priority_list_maharashtra.csv — Maharashtra only
  outputs/priority/top50_portfolio.csv           — optimal 50-segment intervention portfolio

Each row represents one road segment with:
  - Priority rank (global + regional)
  - Road name, GPS coordinates, region
  - Posted limit, 85th-pct speed, Safe System threshold, excess
  - Final score (+ confidence interval)
  - Nilsson fatality reduction (%)
  - Annual economic value of intervention (USD)
  - Archetype + recommended intervention
  - YOLO VRU presence (if available)

Usage:
    python scripts/generate_priority_list.py
    python scripts/generate_priority_list.py --top 100  # export top-N only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import pandas as pd
import numpy as np

from src.config import SCORES_DIR, OUTPUTS


PRIORITY_DIR = OUTPUTS / "priority"

OUTPUT_COLUMNS = [
    # Identity
    "priority_rank_global",
    "priority_rank_region",
    "region",
    "OBJECTID",
    "road_name",
    "road_class",
    "land_use",
    "archetype",
    # Location
    "latitude",
    "longitude",
    # Speed profile
    "posted_limit_kmh",
    "speed_85th_pct_kmh",
    "safe_system_threshold_kmh",
    "speed_excess_kmh",
    "pct_vehicles_over_limit",
    # Scores
    "final_score",
    "final_grade",
    "score_ci_low",
    "score_ci_high",
    "score_std",
    "grade_uncertain",
    # Impact
    "nilsson_fatality_reduction_pct",
    "economic_benefit_usd_annual",
    "economic_benefit_m_annual",
    "impact_efficiency_index",
    # Visual (VLM)
    "vlm_pedestrian_infra",
    "vlm_cyclist_infra",
    "vlm_roadside_activity",
    "vlm_vru_exposure",
    "vlm_road_condition",
    "vlm_signage_quality",
    # Object detection
    "yolo_pedestrian_count",
    "yolo_moto_count",
    "yolo_vru_ratio",
    # Intervention
    "primary_intervention",
    "secondary_intervention",
]


def _extract_centroid(geom):
    """Get centroid lat/lon from a geometry."""
    try:
        c = geom.centroid
        return round(c.y, 6), round(c.x, 6)
    except Exception:
        return None, None


def _build_row(row: pd.Series) -> dict:
    lat, lon = _extract_centroid(row.get("geometry"))
    name = row.get("english_ro") or row.get("names_primary") or ""
    nilsson = row.get("nilsson_reduction_factor", 0) or 0

    return {
        "priority_rank_global":       row.get("priority_rank_global"),
        "priority_rank_region":       row.get("priority_rank_region"),
        "region":                     row.get("region", ""),
        "OBJECTID":                   row.get("OBJECTID", ""),
        "road_name":                  str(name)[:80],
        "road_class":                 row.get("RoadClass", ""),
        "land_use":                   row.get("LandUse", ""),
        "archetype":                  row.get("archetype_name", ""),
        "latitude":                   lat,
        "longitude":                  lon,
        "posted_limit_kmh":           row.get("SpeedLimit"),
        "speed_85th_pct_kmh":         row.get("F85thPercentileSpeed"),
        "safe_system_threshold_kmh":  row.get("safe_system_threshold_kmh"),
        "speed_excess_kmh":           row.get("speed_excess_kmh"),
        "pct_vehicles_over_limit":    round(float(row.get("PercentOverLimit", 0) or 0) * 100, 1),
        "final_score":                row.get("final_score"),
        "final_grade":                row.get("final_grade"),
        "score_ci_low":               row.get("score_ci_low"),
        "score_ci_high":              row.get("score_ci_high"),
        "score_std":                  row.get("score_std"),
        "grade_uncertain":            row.get("grade_uncertain", False),
        "nilsson_fatality_reduction_pct": round(float(nilsson) * 100, 1),
        "economic_benefit_usd_annual":    row.get("economic_benefit_usd"),
        "economic_benefit_m_annual":      row.get("economic_benefit_m"),
        "impact_efficiency_index":        row.get("impact_efficiency_index"),
        "vlm_pedestrian_infra":       row.get("pedestrian_infra"),
        "vlm_cyclist_infra":          row.get("cyclist_infra"),
        "vlm_roadside_activity":      row.get("roadside_activity"),
        "vlm_vru_exposure":           row.get("vru_exposure"),
        "vlm_road_condition":         row.get("road_condition"),
        "vlm_signage_quality":        row.get("signage_quality"),
        "yolo_pedestrian_count":      row.get("yolo_pedestrian_count"),
        "yolo_moto_count":            row.get("yolo_moto_count"),
        "yolo_vru_ratio":             row.get("yolo_vru_ratio"),
        "primary_intervention":       row.get("archetype_intervention", ""),
        "secondary_intervention":     row.get("archetype_secondary", ""),
    }


def load_final_gdfs(region: str) -> gpd.GeoDataFrame:
    """Load final GeoJSON(s) for given region, return merged GeoDataFrame."""
    if region == "all":
        paths = [
            SCORES_DIR / "combined_final.geojson",
        ]
        # Fall back to individual files if combined not available
        if not paths[0].exists():
            paths = [
                SCORES_DIR / "maharashtra_final.geojson",
                SCORES_DIR / "thailand_final.geojson",
            ]
    else:
        paths = [SCORES_DIR / f"{region}_final.geojson"]

    gdfs = []
    for p in paths:
        if p.exists():
            gdfs.append(gpd.read_file(p))
        else:
            print(f"  Warning: {p} not found, skipping")

    if not gdfs:
        return None

    if len(gdfs) == 1:
        return gdfs[0]

    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
    )


def generate_priority_csv(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    grade_filter=("C", "D", "E"),
    top_n: int = None,
):
    """Export a priority CSV for segments matching grade_filter."""
    grade_col = "final_grade" if "final_grade" in gdf.columns else "score_grade"
    mask = gdf[grade_col].isin(grade_filter)
    risk_gdf = gdf[mask].copy()

    # Sort by priority rank if available, else by impact efficiency
    sort_col = "priority_rank_global" if "priority_rank_global" in risk_gdf.columns else "impact_efficiency_index"
    asc = sort_col == "priority_rank_global"
    risk_gdf = risk_gdf.sort_values(sort_col, ascending=asc, na_position="last")

    if top_n:
        risk_gdf = risk_gdf.head(top_n)

    rows = [_build_row(row) for _, row in risk_gdf.iterrows()]
    df = pd.DataFrame(rows)

    # Keep only columns that exist + have data
    existing_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[existing_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  {len(df):,} segments → {out_path}")
    return df


def run(top_n: int = None):
    PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Generating Priority Intervention Lists")
    print("="*60)

    # ── All regions combined ────────────────────────────────────────────────
    gdf_all = load_final_gdfs("all")
    if gdf_all is not None:
        print("\nAll regions (Grade C/D/E):")
        generate_priority_csv(
            gdf_all,
            PRIORITY_DIR / "priority_list_all.csv",
            grade_filter=("C", "D", "E"),
            top_n=top_n,
        )

        print("\nGrade D/E only (Unsafe + Critical):")
        generate_priority_csv(
            gdf_all,
            PRIORITY_DIR / "priority_list_DE_only.csv",
            grade_filter=("D", "E"),
            top_n=top_n,
        )

        print("\nTop-50 portfolio (optimal intervention selection):")
        generate_priority_csv(
            gdf_all,
            PRIORITY_DIR / "top50_portfolio.csv",
            grade_filter=("C", "D", "E"),
            top_n=50,
        )

        print("\nTop-10 portfolio (emergency priority):")
        generate_priority_csv(
            gdf_all,
            PRIORITY_DIR / "top10_emergency.csv",
            grade_filter=("D", "E"),
            top_n=10,
        )

    # ── Per-region files ───────────────────────────────────────────────────
    for region in ["maharashtra", "thailand"]:
        path = SCORES_DIR / f"{region}_final.geojson"
        if not path.exists():
            continue
        gdf_r = gpd.read_file(path)
        print(f"\n{region.title()} (Grade C/D/E):")
        generate_priority_csv(
            gdf_r,
            PRIORITY_DIR / f"priority_list_{region}.csv",
            grade_filter=("C", "D", "E"),
            top_n=top_n,
        )

    # ── Economic summary ───────────────────────────────────────────────────
    if gdf_all is not None and "economic_benefit_usd" in gdf_all.columns:
        grade_col = "final_grade" if "final_grade" in gdf_all.columns else "score_grade"
        de = gdf_all[gdf_all[grade_col].isin(["D", "E"])]
        c_segs = gdf_all[gdf_all[grade_col] == "C"]
        total_de_eco = de["economic_benefit_usd"].sum() / 1e6
        total_c_eco = c_segs["economic_benefit_usd"].sum() / 1e6

        print(f"\n{'='*60}")
        print("ECONOMIC IMPACT SUMMARY")
        print(f"{'='*60}")
        print(f"Grade D/E segments: {len(de):,} | "
              f"Annual eco. value: ${total_de_eco:.1f}M")
        print(f"Grade C segments:   {len(c_segs):,} | "
              f"Annual eco. value (if acted now): ${total_c_eco:.1f}M")
        if len(de):
            top_seg = de.nlargest(1, "economic_benefit_usd").iloc[0]
            name = top_seg.get("english_ro") or top_seg.get("names_primary") or "Unnamed"
            print(f"Highest-value single intervention: ${top_seg['economic_benefit_usd']:,.0f}/yr")
            print(f"  → {name} ({top_seg.get('RoadClass','?')} {top_seg.get('LandUse','?')}, "
                  f"Nilsson {top_seg['nilsson_reduction_factor']*100:.1f}%)")
        print(f"{'='*60}")

    print("\nAll priority lists saved to outputs/priority/")
    print("Next: python scripts/generate_policy_briefs.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=None,
                        help="Limit output to top-N segments per file (default: all)")
    args = parser.parse_args()
    run(args.top)
