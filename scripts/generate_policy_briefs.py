"""
Auto-generate per-segment engineering policy briefs using the Claude API.

For every Grade D and E segment, generates a 250-word professional road safety
engineering assessment tailored to the segment's specific data:
  - Risk profile (what makes it dangerous)
  - Quantified impact (Nilsson estimate, economic value)
  - Three prioritised interventions
  - Urgency classification

Output:
  outputs/policy_briefs/individual/  — one .md file per segment
  outputs/policy_briefs/combined_briefs_thailand.md   — all Thailand Grade D/E in one doc
  outputs/policy_briefs/combined_briefs_maharashtra.md
  outputs/policy_briefs/executive_summary.md          — top-10 global with full briefs

Requirements:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
  python scripts/generate_policy_briefs.py                         # all Grade D/E
  python scripts/generate_policy_briefs.py --region thailand       # Thailand only
  python scripts/generate_policy_briefs.py --grade E               # Critical only
  python scripts/generate_policy_briefs.py --top 50                # top-50 by priority
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import pandas as pd
import numpy as np

from src.config import SCORES_DIR, OUTPUTS


BRIEFS_DIR = OUTPUTS / "policy_briefs"
INDIVIDUAL_DIR = BRIEFS_DIR / "individual"

BRIEF_PROMPT = """You are a senior road safety engineer writing a professional intervention brief for a government transport ministry.

Write a 250-word engineering assessment for the road segment below. Be specific, technical, and immediately actionable.

Structure exactly as follows:
**[Road name or descriptor] — [Region], [Road Class] [Land Use]**

**Risk Profile**
One paragraph. Describe what makes this segment dangerous: the speed mismatch, road type, land use context, what the VLM imagery analysis reveals about infrastructure and road user exposure.

**Quantified Impact**
One paragraph. State: current 85th-percentile speed, posted limit, Safe System threshold, speed excess. State the Nilsson Power Model estimate (fatality reduction if corrected). State the estimated annual economic value of the intervention. If grade is uncertain (CI given), acknowledge it.

**Recommended Interventions**
Three numbered interventions, each one sentence, in priority order from most to least urgent:
1. Immediate (0-3 months)
2. Short-term (3-12 months)
3. Medium-term (1-3 years)

**Urgency: [IMMEDIATE / HIGH / MEDIUM]** — one sentence justification.

---
SEGMENT DATA:
{segment_data}
---

Write the brief now. Use exact numbers from the data. Do not add caveats about data quality."""


def _format_segment_data(row: pd.Series) -> str:
    """Format segment data as a clean JSON string for the prompt."""
    name = row.get("english_ro") or row.get("names_primary") or "Unnamed segment"
    nilsson = row.get("nilsson_reduction_factor", 0) or 0
    eco = row.get("economic_benefit_usd", 0) or 0
    pct_over = row.get("PercentOverLimit", 0) or 0
    score_std = row.get("score_std")

    # VLM features
    vlm_summary = {}
    for feat in ["pedestrian_infra", "cyclist_infra", "roadside_activity",
                 "road_condition", "signage_quality", "vru_exposure", "visibility_quality"]:
        v = row.get(feat)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vlm_summary[feat] = round(float(v), 2)

    # YOLO
    yolo_summary = {}
    for feat in ["yolo_pedestrian_count", "yolo_moto_count", "yolo_vru_ratio"]:
        v = row.get(feat)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            yolo_summary[feat] = round(float(v), 3)

    data = {
        "road_name":               str(name),
        "region":                  str(row.get("region", "")),
        "road_class":              str(row.get("RoadClass", "")),
        "land_use":                str(row.get("LandUse", "")),
        "latitude":                round(float(row.get("lat_mid", 0) or 0), 5),
        "longitude":               round(float(row.get("lon_mid", 0) or 0), 5),
        "posted_limit_kmh":        row.get("SpeedLimit"),
        "speed_85th_pct_kmh":      row.get("F85thPercentileSpeed"),
        "safe_system_threshold_kmh": row.get("safe_system_threshold_kmh"),
        "speed_excess_kmh":        row.get("speed_excess_kmh"),
        "pct_vehicles_over_limit": f"{pct_over*100:.1f}%",
        "final_score":             row.get("final_score"),
        "final_grade":             row.get("final_grade"),
        "score_95pct_ci":          f"{row.get('score_ci_low','?'):.1f}–{row.get('score_ci_high','?'):.1f}"
                                   if (row.get("score_ci_low") and
                                       not (isinstance(row.get("score_ci_low"), float) and
                                            np.isnan(row.get("score_ci_low")))) else "not computed",
        "grade_uncertain":         bool(row.get("grade_uncertain", False)),
        "nilsson_fatality_reduction": f"{nilsson*100:.1f}%",
        "economic_benefit_usd_annual": f"${eco:,.0f}",
        "archetype":               row.get("archetype_name", "Not classified"),
        "priority_rank_region":    int(row.get("priority_rank_region", 0) or 0),
        "vlm_visual_analysis":     vlm_summary if vlm_summary else "not available",
        "object_detection":        yolo_summary if yolo_summary else "not available",
        "primary_intervention":    str(row.get("archetype_intervention", "")),
    }
    return json.dumps(data, indent=2, default=str)


def _generate_brief(client, row: pd.Series, model: str = "claude-opus-4-8") -> str:
    """Generate a brief for one segment using the Anthropic API."""
    segment_data = _format_segment_data(row)
    prompt = BRIEF_PROMPT.format(segment_data=segment_data)

    message = client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _safe_filename(row: pd.Series, idx: int) -> str:
    """Generate a safe filename from segment metadata."""
    region = str(row.get("region", "unknown"))
    obj_id = str(row.get("OBJECTID", idx))
    grade = str(row.get("final_grade", "?"))
    return f"{region}_{grade}_{obj_id.zfill(6)}.md"


def run(region: str, grades: list, top_n: int = None, dry_run: bool = False):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load GeoJSONs
    if region == "all":
        regions = ["maharashtra", "thailand"]
    else:
        regions = [region]

    all_rows = []
    gdfs = {}
    for r in regions:
        path = SCORES_DIR / f"{r}_final.geojson"
        if not path.exists():
            print(f"  Missing: {path}, skipping")
            continue
        gdf = gpd.read_file(path)
        gdfs[r] = gdf
        grade_col = "final_grade" if "final_grade" in gdf.columns else "score_grade"
        mask = gdf[grade_col].isin(grades)
        risk_segs = gdf[mask]
        for _, row in risk_segs.iterrows():
            all_rows.append((r, row))

    # Sort by priority rank
    def _rank_key(x):
        rank = x[1].get("priority_rank_global", 99999) or 99999
        try:
            return float(rank)
        except (TypeError, ValueError):
            return 99999.0

    all_rows.sort(key=_rank_key)

    if top_n:
        all_rows = all_rows[:top_n]

    print(f"\n{'='*60}")
    print(f"Generating {len(all_rows)} policy briefs "
          f"(grades: {','.join(grades)}, model: claude-opus-4-8)")
    print(f"{'='*60}")

    if dry_run:
        print("DRY RUN — showing first segment prompt only\n")
        if all_rows:
            _, row = all_rows[0]
            print(_format_segment_data(row))
        return

    by_region = {r: [] for r in regions}
    all_briefs = []

    for i, (r, row) in enumerate(all_rows, 1):
        name = row.get("english_ro") or row.get("names_primary") or f"Segment {row.get('OBJECTID','?')}"
        grade = row.get("final_grade", "?")
        rank = row.get("priority_rank_region", "?")
        print(f"[{i:3d}/{len(all_rows)}] Grade {grade} | Rank #{rank} | {r.title()} | {name[:45]}")

        try:
            brief = _generate_brief(client, row)
        except Exception as e:
            print(f"  ERROR: {e}")
            brief = f"Brief generation failed: {e}"

        # Save individual file
        fname = _safe_filename(row, i)
        fpath = INDIVIDUAL_DIR / fname
        with open(fpath, "w") as f:
            f.write(f"<!-- OBJECTID: {row.get('OBJECTID','')} | "
                    f"Region: {r} | Grade: {grade} | "
                    f"Priority: #{rank} -->\n\n")
            f.write(brief)
            f.write("\n")

        by_region[r].append(brief)
        all_briefs.append((r, grade, rank, name, brief))

        # Respect API rate limits
        if i % 10 == 0:
            time.sleep(1)

    # ── Combined regional files ────────────────────────────────────────────
    for r, briefs in by_region.items():
        if not briefs:
            continue
        combined_path = BRIEFS_DIR / f"combined_briefs_{r}.md"
        with open(combined_path, "w") as f:
            f.write(f"# Road Safety Policy Briefs — {r.title()}\n")
            f.write(f"*Generated by AI for Safer Roads pipeline | "
                    f"Grade D/E segments | {len(briefs)} segments*\n\n")
            f.write("---\n\n")
            for brief in briefs:
                f.write(brief)
                f.write("\n\n---\n\n")
        print(f"\nCombined briefs → {combined_path}")

    # ── Executive summary (top-10 globally) ───────────────────────────────
    exec_path = BRIEFS_DIR / "executive_summary.md"
    with open(exec_path, "w") as f:
        f.write("# Executive Summary: Top Priority Road Safety Interventions\n")
        f.write("*AI for Safer Roads Innovation Challenge 2026 | "
                "hksamm | Pusan National University*\n\n")
        f.write("This document presents the ten highest-priority road safety "
                "interventions identified by the Speed Safety Score pipeline "
                "across Thailand and Maharashtra, India. Segments are ranked by "
                "impact efficiency index: annual economic value of the intervention "
                "relative to the estimated review and implementation cost.\n\n")
        f.write("---\n\n")
        top10 = sorted(all_briefs, key=lambda x: float(x[2] or 99999))[:10]
        for pos, (r, grade, rank, name, brief) in enumerate(top10, 1):
            f.write(f"## #{pos} — {name}\n")
            f.write(f"*{r.title()} | Grade {grade} | Regional rank #{rank}*\n\n")
            f.write(brief)
            f.write("\n\n---\n\n")
    print(f"Executive summary → {exec_path}")

    print(f"\n{'='*60}")
    print(f"Done: {len(all_briefs)} briefs generated")
    print(f"Individual: {INDIVIDUAL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="all",
                        choices=["maharashtra", "thailand", "all"])
    parser.add_argument("--grade", default="D,E",
                        help="Comma-separated grades e.g. D,E or E")
    parser.add_argument("--top", type=int, default=None,
                        help="Generate briefs for top-N segments only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first segment prompt without calling API")
    args = parser.parse_args()
    grades = [g.strip().upper() for g in args.grade.split(",")]
    run(args.region, grades, args.top, args.dry_run)
