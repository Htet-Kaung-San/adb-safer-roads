"""
Budget-Constrained Intervention Portfolio Optimiser.

Given a finite number of road safety interventions a government can fund
this year, which segments should be prioritised to maximise lives saved
and economic value?

Method: greedy fractional knapsack (optimal for linear value/cost ratios).

Each segment has:
    value  = economic_benefit_usd × nilsson_reduction_factor   (expected annual lives saved × $)
    cost   = review_cost_index (proxy: higher-class / longer roads cost more to review)

The efficiency ratio = value / cost ranks segments for selection.

Budget scenarios are pre-computed for 10 / 25 / 50 / 100 / 250 / 500 segments.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict


# ── Review cost proxy ─────────────────────────────────────────────────────
# Higher road class = more stakeholders, more traffic management needed = higher cost.
# Longer segment = more physical works.
# These are relative weights, not actual dollars.
ROAD_CLASS_COST_WEIGHT = {
    "motorway":  3.0,
    "trunk":     2.0,
    "primary":   1.5,
    "secondary": 1.0,
    None:        1.2,
}

LENGTH_COST_WEIGHT_PER_KM = 0.05   # 1 km adds 0.05 to cost index
MAX_LENGTH_KM_FOR_COST = 20.0      # cap at 20 km


def _review_cost_index(row: pd.Series) -> float:
    """
    Proxy for the cost/difficulty of reviewing and implementing a speed limit change.
    Higher-class, longer, urban roads cost more to change.
    Returns a value nominally around 1.0–4.0.
    """
    road_class = str(row.get("RoadClass", "")).lower() if row.get("RoadClass") else None
    base = ROAD_CLASS_COST_WEIGHT.get(road_class, 1.2)

    length_km = float(row.get("length_km", 1.0) or 1.0)
    length_cost = min(length_km, MAX_LENGTH_KM_FOR_COST) * LENGTH_COST_WEIGHT_PER_KM

    # Urban roads require more coordination (traffic, residents, business access)
    urban_multiplier = 1.2 if str(row.get("LandUse", "")).upper() == "URBAN" else 1.0

    return (base + length_cost) * urban_multiplier


def compute_priority_ranking(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add priority ranking and efficiency metrics to the GeoDataFrame.

    Computes:
        review_cost_index          — proxy review/intervention cost
        impact_efficiency_index    — economic_benefit / review_cost (value density)
        lives_per_cost_index       — nilsson_reduction × traffic / review_cost
        priority_rank_global       — rank across both regions combined (1 = highest priority)
        priority_rank_region       — rank within region

    Ranking is by impact_efficiency_index descending (most lives saved per unit cost first).
    Only assigns ranks to Grade C/D/E segments — A/B segments get rank=NaN.
    """
    gdf = gdf.copy()

    # Review cost index per segment
    gdf["review_cost_index"] = gdf.apply(_review_cost_index, axis=1).round(3)

    # Economic value of intervention
    eco_col = "economic_benefit_usd"
    if eco_col not in gdf.columns:
        gdf[eco_col] = 0

    # Impact efficiency: dollars saved per unit of review cost
    cost = gdf["review_cost_index"].clip(lower=0.1)
    gdf["impact_efficiency_index"] = (gdf[eco_col] / cost).round(2)

    # Lives-per-cost: nilsson × traffic exposure / cost
    nilsson = gdf.get("nilsson_reduction_factor", pd.Series(0.0, index=gdf.index)).fillna(0)
    rp = gdf["RankedPercentile"].fillna(0)
    rp_max = rp.max()
    rp_norm = (rp / rp_max).clip(0, 1) if rp_max > 0 else pd.Series(0.0, index=gdf.index)
    gdf["lives_per_cost_index"] = ((nilsson * rp_norm) / cost).round(4)

    # Grade filter for ranking
    grade_col = "final_grade" if "final_grade" in gdf.columns else "score_grade"
    rank_mask = gdf[grade_col].isin(["C", "D", "E"])

    # Global rank (across all regions in this GeoDataFrame)
    ranked_indices = (
        gdf[rank_mask]
        .sort_values("impact_efficiency_index", ascending=False)
        .index
    )
    gdf["priority_rank_global"] = np.nan
    for rank_pos, idx in enumerate(ranked_indices, start=1):
        gdf.at[idx, "priority_rank_global"] = rank_pos

    # Per-region rank
    gdf["priority_rank_region"] = np.nan
    if "region" in gdf.columns:
        for region in gdf["region"].unique():
            region_mask = rank_mask & (gdf["region"] == region)
            region_ranked = (
                gdf[region_mask]
                .sort_values("impact_efficiency_index", ascending=False)
                .index
            )
            for rank_pos, idx in enumerate(region_ranked, start=1):
                gdf.at[idx, "priority_rank_region"] = rank_pos

    return gdf


def build_portfolio_scenarios(
    gdf: gpd.GeoDataFrame,
    budgets: List[int] = None,
) -> Dict[int, Dict]:
    """
    For each budget scenario (number of segments to address), compute:
        - which segments to select (greedy by impact_efficiency_index)
        - total economic benefit
        - total estimated fatal crash reduction (proxy)
        - grade distribution of selected segments

    Returns dict of {budget_n: {segments: GeoDataFrame, stats: dict}}.
    """
    if budgets is None:
        budgets = [10, 25, 50, 100, 250, 500]

    grade_col = "final_grade" if "final_grade" in gdf.columns else "score_grade"
    rank_mask = gdf[grade_col].isin(["C", "D", "E"])
    ranked = gdf[rank_mask].sort_values("impact_efficiency_index", ascending=False).copy()

    portfolios = {}
    for n in budgets:
        selected = ranked.head(n).copy()
        eco = selected.get("economic_benefit_usd", pd.Series(0)).sum()
        nilsson_avg = selected.get("nilsson_reduction_factor", pd.Series(0)).mean()
        grade_dist = selected[grade_col].value_counts().to_dict()

        portfolios[n] = {
            "segments":              selected,
            "n_selected":            len(selected),
            "total_economic_usd":    int(eco),
            "total_economic_m":      round(eco / 1e6, 2),
            "avg_nilsson_reduction": round(float(nilsson_avg) * 100, 1),
            "grade_distribution":    grade_dist,
            "regions":               selected.get("region", pd.Series()).value_counts().to_dict()
                                     if "region" in selected.columns else {},
        }

    return portfolios


def print_portfolio_summary(portfolios: Dict[int, Dict]):
    """Print a formatted summary table of all portfolio scenarios."""
    print("\n" + "=" * 75)
    print("INTERVENTION PORTFOLIO SCENARIOS — Greedy Optimisation")
    print("=" * 75)
    print(f"{'Budget':>8}  {'Selected':>9}  {'Eco. Benefit':>13}  "
          f"{'Avg Nilsson':>12}  {'Grade D+E':>10}")
    print("-" * 75)
    for n, p in sorted(portfolios.items()):
        grade_de = p["grade_distribution"].get("D", 0) + p["grade_distribution"].get("E", 0)
        print(f"{n:>8}  {p['n_selected']:>9}  "
              f"${p['total_economic_m']:>11.2f}M  "
              f"{p['avg_nilsson_reduction']:>11.1f}%  "
              f"{grade_de:>10}")
    print("=" * 75)
