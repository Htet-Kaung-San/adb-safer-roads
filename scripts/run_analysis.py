"""
Post-GNN Analysis Pipeline — runs on Mac after syncing final GeoJSONs from server.

Adds to each final GeoJSON (in-place update):
    1. Economic impact analysis  (economic_benefit_usd, annual_vmt_proxy, ...)
    2. Segment archetype clustering  (archetype_name, description, intervention, ...)
    3. Priority ranking & intervention optimiser  (priority_rank_global/region, ...)

Then prints:
    - Portfolio scenario table (top 10 / 25 / 50 / 100 / 500 interventions)
    - Archetype summary per region
    - Combined cross-region rankings

Usage:
    python scripts/run_analysis.py                    # both regions
    python scripts/run_analysis.py --region thailand  # single region

Outputs (updated in-place):
    outputs/scores/maharashtra_final.geojson
    outputs/scores/thailand_final.geojson

Also writes:
    outputs/scores/combined_final.geojson   (both regions merged, globally ranked)
    outputs/analysis/archetype_summary.csv
    outputs/analysis/portfolio_scenarios.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import pandas as pd
import numpy as np

from src.scoring.safe_system import compute_economic_impact
from src.analysis.clustering import assign_archetypes, get_archetype_summary
from src.analysis.optimizer import compute_priority_ranking, build_portfolio_scenarios, print_portfolio_summary
from src.config import SCORES_DIR, OUTPUTS


ANALYSIS_DIR = OUTPUTS / "analysis"


def run(region: str):
    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_gdfs = []

    for r in regions:
        print(f"\n{'='*65}")
        print(f"Post-GNN Analysis: {r.title()}")
        print('='*65)

        final_path = SCORES_DIR / f"{r}_final.geojson"
        if not final_path.exists():
            print(f"Missing: {final_path}. Run run_stage3_gnn.py on the server first.")
            continue

        gdf = gpd.read_file(final_path)
        print(f"Loaded {len(gdf):,} segments")

        # ── 1. Economic impact ──────────────────────────────────────────────
        print("\n[1/3] Computing economic impact (VOSL × Nilsson × VMT proxy) …")
        gdf = compute_economic_impact(gdf)

        eco_risk = gdf[gdf["final_grade"].isin(["C", "D", "E"])]
        if len(eco_risk):
            total_eco_m = eco_risk["economic_benefit_usd"].sum() / 1e6
            max_eco = eco_risk.nlargest(1, "economic_benefit_usd").iloc[0]
            print(f"  Total economic value (Grade C/D/E): ${total_eco_m:.1f}M / year")
            print(f"  Highest-value segment: ${max_eco['economic_benefit_usd']:,.0f}/year "
                  f"(Grade {max_eco['final_grade']}, "
                  f"{max_eco.get('RoadClass','?')} {max_eco.get('LandUse','?')})")

        # ── 2. Archetype clustering ─────────────────────────────────────────
        print("\n[2/3] Clustering Grade C/D/E segments into risk archetypes …")
        gdf = assign_archetypes(gdf, n_clusters=5)

        arch_summary = get_archetype_summary(gdf)
        if not arch_summary.empty:
            print(f"\n  Archetype summary for {r.title()}:")
            for _, row in arch_summary.iterrows():
                print(f"    {row['icon']} {row['archetype']}: {row['count']} segments | "
                      f"avg score {row['avg_final_score']} | "
                      f"${row['total_economic_benefit_m']:.2f}M total")

        # ── 3. Priority ranking ─────────────────────────────────────────────
        print("\n[3/3] Computing priority ranking (impact efficiency) …")
        gdf = compute_priority_ranking(gdf)

        top10 = gdf[gdf["final_grade"].isin(["C", "D", "E"])].nsmallest(10, "priority_rank_region")
        if len(top10):
            print(f"\n  Top 10 priority segments in {r.title()}:")
            for _, row in top10.iterrows():
                name = row.get("english_ro") or row.get("names_primary") or "Unnamed"
                print(f"    #{row['priority_rank_region']:.0f} | "
                      f"Grade {row['final_grade']} | "
                      f"{row.get('RoadClass','?')} {row.get('LandUse','?')} | "
                      f"{name[:35]} | "
                      f"${row['economic_benefit_usd']:,.0f}/yr | "
                      f"Nilsson: {row['nilsson_reduction_factor']*100:.1f}%")

        # ── Save updated GeoJSON ────────────────────────────────────────────
        gdf.to_file(final_path, driver="GeoJSON")
        print(f"\n  Updated GeoJSON → {final_path}")

        all_gdfs.append(gdf)

        # ── Per-region archetype summary CSV ──────────────────────────────
        arch_out = ANALYSIS_DIR / f"{r}_archetype_summary.csv"
        if not arch_summary.empty:
            arch_summary.to_csv(arch_out, index=False)
            print(f"  Archetype summary → {arch_out}")

    # ── Combined cross-region analysis ─────────────────────────────────────
    if len(all_gdfs) == 2:
        print(f"\n{'='*65}")
        print("Combined Cross-Region Analysis")
        print('='*65)

        combined_gdf = gpd.GeoDataFrame.from_features(
            [f for gdf in all_gdfs for f in gdf.__geo_interface__["features"]],
            crs=all_gdfs[0].crs,
        )

        # Global priority ranking across both regions
        combined_gdf = compute_priority_ranking(combined_gdf)

        combined_path = SCORES_DIR / "combined_final.geojson"
        combined_gdf.to_file(combined_path, driver="GeoJSON")
        print(f"Combined GeoJSON → {combined_path}")

        # Portfolio scenarios on combined dataset
        print("\nPortfolio optimisation (combined regions):")
        portfolios = build_portfolio_scenarios(combined_gdf)
        print_portfolio_summary(portfolios)

        # Save portfolio JSON
        port_out = ANALYSIS_DIR / "portfolio_scenarios.json"
        port_serialisable = {}
        for n, p in portfolios.items():
            port_serialisable[str(n)] = {
                k: v for k, v in p.items() if k != "segments"
            }
        with open(port_out, "w") as f:
            json.dump(port_serialisable, f, indent=2)
        print(f"Portfolio scenarios → {port_out}")

        # Combined archetype summary
        combined_arch = get_archetype_summary(combined_gdf)
        if not combined_arch.empty:
            combined_arch_out = ANALYSIS_DIR / "combined_archetype_summary.csv"
            combined_arch.to_csv(combined_arch_out, index=False)
            print(f"Combined archetype summary → {combined_arch_out}")

            print("\nArchetype breakdown (both regions):")
            for _, row in combined_arch.iterrows():
                print(f"  {row['icon']} {row['archetype']:30s} "
                      f"{row['count']:5d} segs | "
                      f"${row['total_economic_benefit_m']:6.2f}M | "
                      f"avg Nilsson {row['avg_nilsson_pct']:.1f}%")

        # Global top-10 segments
        risk_combined = combined_gdf[combined_gdf["final_grade"].isin(["D", "E"])]
        if len(risk_combined):
            print(f"\nTop 10 highest-priority segments globally:")
            top10_global = risk_combined.nsmallest(10, "priority_rank_global")
            for _, row in top10_global.iterrows():
                name = row.get("english_ro") or row.get("names_primary") or "Unnamed"
                print(f"  #{row['priority_rank_global']:.0f} | "
                      f"{row.get('region','?').title():12s} | "
                      f"Grade {row['final_grade']} | "
                      f"Score {row['final_score']:.1f} ± {row.get('score_std', float('nan')):.1f} | "
                      f"{row.get('RoadClass','?')} {row.get('LandUse','?')} | "
                      f"${row['economic_benefit_usd']:,.0f}/yr | "
                      f"Nilsson: {row['nilsson_reduction_factor']*100:.1f}%")

    print(f"\n{'='*65}")
    print("Analysis complete. Now run:")
    print("  python scripts/generate_priority_list.py")
    print("  python scripts/generate_maps.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all",
                        choices=["maharashtra", "thailand", "all"])
    args = parser.parse_args()
    run(args.region)
