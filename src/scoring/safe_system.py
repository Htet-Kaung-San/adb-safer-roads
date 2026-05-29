"""
Stage 1: Safe System Tabular Scorer.

Produces a Speed Safety Score (0-100) per road segment based on:
  - Speed deviation from WHO Safe System thresholds
  - Posted limit excess above Safe System threshold
  - Speeding prevalence (% vehicles over limit)
  - Traffic exposure (weighted sample size)
  - VRU vulnerability (inverse helmet SPI)

Score bands: A (Safe) → B (Adequate) → C (Caution) → D (Unsafe) → E (Critical)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional

from src.config import (
    SAFE_SYSTEM_THRESHOLDS,
    HELMET_SPI,
    HELMET_BASELINE,
    SCORE_WEIGHTS,
    SCORE_BANDS,
    NILSSON_EXPONENTS,
)


def get_safe_system_threshold(road_class: Optional[str], land_use: Optional[str]) -> float:
    """Return the Safe System speed threshold (km/h) for the given road/context."""
    key = (road_class, land_use)
    if key in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[key])
    # Fallback: try with None for one dimension
    if (road_class, None) in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[(road_class, None)])
    if (None, land_use) in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[(None, land_use)])
    return float(SAFE_SYSTEM_THRESHOLDS[(None, None)])


def _normalise(series: pd.Series, cap: float = 100.0) -> pd.Series:
    """Clip to [0, cap] then scale to [0, 1]."""
    return series.clip(0, cap) / cap


def score_speed_deviation(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    How far does the 85th-percentile operating speed exceed the Safe System threshold?

    A road where cars typically travel at 97 km/h on a 80 km/h-threshold primary road
    gets a high sub-score regardless of what the posted limit says.
    Excess capped at 60 km/h for normalisation (beyond that, uniformly critical).
    """
    thresholds = gdf.apply(
        lambda r: get_safe_system_threshold(r.get("RoadClass"), r.get("LandUse")),
        axis=1,
    )
    excess = (gdf["F85thPercentileSpeed"] - thresholds).clip(lower=0)
    return _normalise(excess, cap=60.0)


def score_posted_limit_excess(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Is the *posted* speed limit itself already above the Safe System threshold?
    This captures legislatively unsafe limits even when drivers respect them.
    Capped at 50 km/h excess.
    """
    thresholds = gdf.apply(
        lambda r: get_safe_system_threshold(r.get("RoadClass"), r.get("LandUse")),
        axis=1,
    )
    excess = (gdf["SpeedLimit"] - thresholds).clip(lower=0)
    return _normalise(excess, cap=50.0)


def score_speeding_prevalence(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Fraction of vehicles exceeding the posted speed limit.
    PercentOverLimit is already 0–1 in the data.
    """
    return gdf["PercentOverLimit"].clip(0, 1).fillna(0)


def score_traffic_exposure(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Traffic volume proxy: more vehicles exposed → more lives at risk.
    Normalises RankedPercentile to [0,1] within the dataset, since
    Maharashtra uses 0-1 scale and Thailand uses 0-100 scale.
    """
    rp = gdf["RankedPercentile"].fillna(0)
    rp_max = rp.max()
    if rp_max <= 0:
        return pd.Series(0.0, index=gdf.index)
    return (rp / rp_max).clip(0, 1)


def score_vru_vulnerability(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Vulnerable Road User vulnerability: inverse of helmet-wearing rate.
    Lower helmet use → higher vulnerability → higher score contribution.

    Uses regional SPI (Maharashtra vs Thailand) mapped by 'region' column.
    """
    def _vru(region: str) -> float:
        spi = HELMET_SPI.get(region, HELMET_BASELINE)
        # Normalise: 0 (everyone helmeted at baseline) → 1 (nobody helmeted)
        return max(0.0, (HELMET_BASELINE - spi) / HELMET_BASELINE)

    if "region" in gdf.columns:
        return gdf["region"].map(_vru).fillna(0.5)
    return pd.Series(0.5, index=gdf.index)


def compute_speed_safety_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute the Speed Safety Score for every segment in gdf.

    Adds columns:
        sub_speed_deviation, sub_posted_excess, sub_speeding_prevalence,
        sub_traffic_exposure, sub_vru_vulnerability,
        speed_safety_score  (0–100),
        score_grade         ('A'–'E'),
        score_label         ('Safe' … 'Critical'),
        safe_system_threshold_kmh,
        speed_excess_kmh    (F85th − threshold, clipped to 0)

    Returns the same GeoDataFrame with new columns appended (in-place copy).
    """
    gdf = gdf.copy()
    w = SCORE_WEIGHTS

    sub = {
        "speed_deviation":      score_speed_deviation(gdf),
        "posted_limit_excess":  score_posted_limit_excess(gdf),
        "speeding_prevalence":  score_speeding_prevalence(gdf),
        "traffic_exposure":     score_traffic_exposure(gdf),
        "vru_vulnerability":    score_vru_vulnerability(gdf),
    }

    for key, series in sub.items():
        gdf[f"sub_{key}"] = series.values

    score = sum(sub[k] * w[k] for k in w)
    gdf["speed_safety_score"] = score.clip(0, 100).round(2)

    # ── Thresholds & excess for reporting ─────────────────────────────────
    gdf["safe_system_threshold_kmh"] = gdf.apply(
        lambda r: get_safe_system_threshold(r.get("RoadClass"), r.get("LandUse")),
        axis=1,
    )
    gdf["speed_excess_kmh"] = (
        gdf["F85thPercentileSpeed"] - gdf["safe_system_threshold_kmh"]
    ).clip(lower=0).round(1)

    # ── Grade ─────────────────────────────────────────────────────────────
    grades, labels = [], []
    for score_val in gdf["speed_safety_score"]:
        grade, label = "E", "Critical"
        for lo, hi, g, lbl, _ in SCORE_BANDS:
            if lo <= score_val < hi:
                grade, label = g, lbl
                break
        grades.append(grade)
        labels.append(label)

    gdf["score_grade"] = grades
    gdf["score_label"] = labels

    return gdf


def compute_counterfactual_impact(
    gdf: gpd.GeoDataFrame,
    target_speed_kmh: Optional[float] = None,
    outcome: str = "fatality",
) -> gpd.GeoDataFrame:
    """
    Estimate lives saved if unsafe segments had their speed limit reduced to
    the Safe System threshold (or a custom target_speed_kmh).

    Uses Nilsson's Power Model: fatality_reduction = 1 − (v_new / v_old)^exponent

    Adds columns:
        target_speed_kmh, reduction_factor, annual_exposure_proxy,
        estimated_fatalities_prevented_relative  (unit-free, relative index)

    Note: We don't have crash counts, so this is a *relative* risk reduction
    index useful for ranking interventions, not an absolute prediction.
    """
    gdf = gdf.copy()
    exp = NILSSON_EXPONENTS[outcome]

    v_old = gdf["F85thPercentileSpeed"].clip(lower=1)

    if target_speed_kmh is not None:
        v_new = pd.Series(float(target_speed_kmh), index=gdf.index)
    else:
        v_new = gdf["safe_system_threshold_kmh"].clip(lower=1)

    v_new = v_new.clip(upper=v_old)   # can't increase speed
    reduction = 1.0 - (v_new / v_old) ** exp
    reduction = reduction.clip(0, 1)

    gdf["target_speed_kmh"] = v_new.round(1)
    gdf["nilsson_reduction_factor"] = reduction.round(4)

    # Weight by traffic × road length for a relative exposure-adjusted index
    exposure = (
        gdf["RankedPercentile"].fillna(0) / 100.0 *
        gdf["length_km"].fillna(0)
    )
    gdf["estimated_impact_index"] = (reduction * exposure).round(4)

    return gdf
