"""
Counterfactual Policy Simulation
---------------------------------
Answers: "If Thailand / Maharashtra adopted Safe System speed limits nationally,
how many lives would be saved and what is the economic value over 10 years?"

Two scenarios per region:
  A) Full Safe System adoption — all segments corrected to threshold
  B) Urban primary / trunk only — the single legislative decision that affects the most Grade D roads

Outputs:
  outputs/analysis/counterfactual_summary.json
  outputs/analysis/counterfactual_report.md   ← ready to paste into submission
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import SCORES_DIR, OUTPUTS

ANALYSIS_DIR = OUTPUTS / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
VOSL = {"thailand": 1_260_000, "maharashtra": 420_000}
CRASH_RATE = {"thailand": 8.4, "maharashtra": 11.2}     # per 100M VMT
DAILY_TRAFFIC_BASE = 15_000                              # conservative urban VPD
KM_LENGTH_MEAN = 1.2                                     # avg segment length km
YEARS = 10


def run():
    results = {}

    for region in ["thailand", "maharashtra"]:
        final_path = SCORES_DIR / f"{region}_final.geojson"
        if not final_path.exists():
            print(f"Skipping {region} — no final GeoJSON found")
            continue

        gdf = gpd.read_file(final_path)
        print(f"\n{'='*60}")
        print(f"Counterfactual simulation: {region.title()}")
        print(f"Total segments: {len(gdf):,}")

        # Use already-computed economic values from run_analysis.py
        # crashes_averted_proxy and economic_benefit_usd are in the GeoJSON
        eco_col = "economic_benefit_usd"
        crash_col = "crashes_averted_proxy"

        # Grade to target per region:
        # Thailand has Grade D (Unsafe) — the high-priority tier
        # Maharashtra GNN reclassified all D→C, so we target Grade C
        target_grade = ["D"] if region == "thailand" else ["C"]

        # ── Scenario A: Full Safe System adoption (target grade, all segments) ──
        affected_all = gdf[gdf["final_grade"].isin(target_grade)].copy()
        lives_a = float(affected_all[crash_col].fillna(0).sum()) if crash_col in affected_all.columns else 0
        econ_a  = float(affected_all[eco_col].fillna(0).sum()) if eco_col in affected_all.columns else 0

        # ── Scenario B: Urban primary/trunk reform only ─────────────────
        urban_primary = gdf[
            (gdf["final_grade"].isin(target_grade)) &
            (gdf["LandUse"].str.upper() == "URBAN") &
            (gdf["RoadClass"].str.lower().isin(["primary", "trunk"]))
        ].copy()
        lives_b = float(urban_primary[crash_col].fillna(0).sum()) if crash_col in urban_primary.columns else 0
        econ_b  = float(urban_primary[eco_col].fillna(0).sum()) if eco_col in urban_primary.columns else 0

        results[region] = {
            "scenario_a_full_safe_system": {
                "affected_segments": len(affected_all),
                "annual_crashes_averted_proxy": round(lives_a, 2),
                "10yr_crashes_averted_proxy": round(lives_a * YEARS, 2),
                "annual_economic_value_usd_m": round(econ_a / 1e6, 1),
                "10yr_economic_value_usd_m": round(econ_a * YEARS / 1e6, 1),
            },
            "scenario_b_urban_primary_trunk_only": {
                "affected_segments": len(urban_primary),
                "annual_crashes_averted_proxy": round(lives_b, 2),
                "10yr_crashes_averted_proxy": round(lives_b * YEARS, 2),
                "annual_economic_value_usd_m": round(econ_b / 1e6, 1),
                "10yr_economic_value_usd_m": round(econ_b * YEARS / 1e6, 1),
                "policy_action": "Single national speed limit regulation — no physical works required",
            },
        }

        r = results[region]
        a = r["scenario_a_full_safe_system"]
        b = r["scenario_b_urban_primary_trunk_only"]

        print(f"\n  Scenario A — Full Safe System adoption ({a['affected_segments']} segments):")
        print(f"    Annual crashes averted (proxy): {a['annual_crashes_averted_proxy']:.2f}")
        print(f"    Annual economic value: ${a['annual_economic_value_usd_m']:.1f}M USD")
        print(f"    10-year economic value: ${a['10yr_economic_value_usd_m']:.1f}M USD")

        print(f"\n  Scenario B — Urban primary/trunk reform only ({b['affected_segments']} segments):")
        print(f"    Annual crashes averted (proxy): {b['annual_crashes_averted_proxy']:.2f}")
        print(f"    Annual economic value: ${b['annual_economic_value_usd_m']:.1f}M USD")
        print(f"    10-year economic value: ${b['10yr_economic_value_usd_m']:.1f}M USD")
        print(f"    → Achievable via one regulation, no capital expenditure")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_out = ANALYSIS_DIR / "counterfactual_summary.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCounterfactual summary → {json_out}")

    # ── Generate markdown report ───────────────────────────────────────────────
    md_lines = ["# Counterfactual Policy Simulation\n"]
    md_lines.append(
        "*Based on Nilsson Power Model × VOSL × VMT proxy. "
        "These are relative indices for ranking interventions — not calibrated absolute predictions.*\n"
    )

    for region, r in results.items():
        a = r["scenario_a_full_safe_system"]
        b = r["scenario_b_urban_primary_trunk_only"]
        md_lines.append(f"\n## {region.title()}\n")
        md_lines.append(f"### Scenario A — Full Safe System Adoption\n")
        md_lines.append(f"- **{a['affected_segments']} segments** corrected to Safe System threshold\n")
        md_lines.append(f"- **${a['annual_economic_value_usd_m']:.1f}M USD/year** economic value (VOSL × Nilsson × VMT proxy)\n")
        md_lines.append(f"- **${a['10yr_economic_value_usd_m']:.0f}M USD** in economic value over 10 years\n")
        md_lines.append(f"\n### Scenario B — Urban Primary/Trunk Reform Only (Single Policy Decision)\n")
        md_lines.append(f"- **{b['affected_segments']} segments** — achievable via one national regulation\n")
        md_lines.append(f"- **${b['annual_economic_value_usd_m']:.1f}M USD/year** economic value\n")
        md_lines.append(f"- **${b['10yr_economic_value_usd_m']:.0f}M USD** in economic value over 10 years\n")
        md_lines.append(f"- No capital expenditure required — speed limit change only\n")

    md_out = ANALYSIS_DIR / "counterfactual_report.md"
    with open(md_out, "w") as f:
        f.writelines(md_lines)
    print(f"Counterfactual report → {md_out}")

    return results


if __name__ == "__main__":
    run()
