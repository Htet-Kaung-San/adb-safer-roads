"""
Stage 1: Safe System Tabular Scorer.

Produces a Speed Safety Score (0-100) per road segment based on:
  - Speed deviation from WHO Safe System thresholds
  - Posted limit excess above Safe System threshold
  - Speeding prevalence (% vehicles over limit)
  - Traffic exposure (weighted sample size)
  - VRU vulnerability (inverse helmet SPI)

Score bands: A (Safe) → B (Adequate) → C (Caution) → D (Unsafe) → E (Critical)

Also provides:
  - Nilsson Power Model counterfactual impact (lives saved if limit corrected)
  - Economic impact quantification (USD value of lives saved per segment per year)
"""
import math
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
    VOSL_USD,
    CRASH_RATE_PER_100M_VMT,
    TRAFFIC_VOLUME_MIN_VPD,
    TRAFFIC_VOLUME_MAX_VPD,
)


def get_safe_system_threshold(road_class: Optional[str], land_use: Optional[str]) -> float:
    """Return the Safe System speed threshold (km/h) for the given road/context."""
    key = (road_class, land_use)
    if key in SAFE_SYSTEM_THRESHOLDS:
        return float(SAFE_SYSTEM_THRESHOLDS[key])
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
    Captures legislatively unsafe limits even when drivers respect them.
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
    PercentOverLimit is already 0-1 in the data.
    """
    return gdf["PercentOverLimit"].clip(0, 1).fillna(0)


def score_traffic_exposure(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Traffic volume proxy: more vehicles exposed → more lives at risk.
    Normalises RankedPercentile to [0,1] within the dataset.
    Maharashtra uses 0-1 scale; Thailand uses 0-100 scale.
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
        speed_safety_score  (0-100),
        score_grade         ('A'-'E'),
        score_label         ('Safe' ... 'Critical'),
        safe_system_threshold_kmh,
        speed_excess_kmh    (F85th - threshold, clipped to 0)
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

    gdf["safe_system_threshold_kmh"] = gdf.apply(
        lambda r: get_safe_system_threshold(r.get("RoadClass"), r.get("LandUse")),
        axis=1,
    )
    gdf["speed_excess_kmh"] = (
        gdf["F85thPercentileSpeed"] - gdf["safe_system_threshold_kmh"]
    ).clip(lower=0).round(1)

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
    Estimate fatality reduction if unsafe segments had limits reduced to Safe System threshold.
    Uses Nilsson's Power Model: fatality_reduction = 1 - (v_new / v_old)^exponent

    Adds: target_speed_kmh, nilsson_reduction_factor, estimated_impact_index
    """
    gdf = gdf.copy()
    exp = NILSSON_EXPONENTS[outcome]

    v_old = gdf["F85thPercentileSpeed"].clip(lower=1)

    if target_speed_kmh is not None:
        v_new = pd.Series(float(target_speed_kmh), index=gdf.index)
    else:
        v_new = gdf["safe_system_threshold_kmh"].clip(lower=1)

    v_new = v_new.clip(upper=v_old)
    reduction = 1.0 - (v_new / v_old) ** exp
    reduction = reduction.clip(0, 1)

    gdf["target_speed_kmh"] = v_new.round(1)
    gdf["nilsson_reduction_factor"] = reduction.round(4)

    exposure = (
        gdf["RankedPercentile"].fillna(0) / 100.0 *
        gdf["length_km"].fillna(0)
    )
    gdf["estimated_impact_index"] = (reduction * exposure).round(4)

    return gdf


def _estimate_daily_traffic(rp_normalised: float) -> float:
    """
    Convert normalised RankedPercentile [0,1] to estimated vehicles/day.
    Log-linear interpolation: p=0 → 200 veh/day, p=1 → 60,000 veh/day.
    This is a relative proxy for economic ranking — not a calibrated count.
    """
    log_min = math.log10(TRAFFIC_VOLUME_MIN_VPD)
    log_max = math.log10(TRAFFIC_VOLUME_MAX_VPD)
    return 10 ** (log_min + rp_normalised * (log_max - log_min))


def compute_economic_impact(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Quantify the economic value of correcting speed limits on each segment.

    Method:
        annual_vmt_km  = estimated_daily_traffic × length_km × 365
        crashes_averted = (annual_vmt_km / 1e8) × crash_rate × nilsson_reduction
        economic_benefit_usd = crashes_averted × VOSL

    Requires: nilsson_reduction_factor (from compute_counterfactual_impact),
              RankedPercentile, length_km, region.

    Adds columns:
        annual_vmt_proxy      — estimated annual vehicle-kilometres (relative)
        crashes_averted_proxy — estimated annual fatal crashes averted (relative)
        economic_benefit_usd  — annual economic value of intervention (USD, relative)
        economic_benefit_m    — same, in millions USD
    """
    gdf = gdf.copy()

    # Normalise RankedPercentile to [0,1]
    rp = gdf["RankedPercentile"].fillna(0)
    rp_max = rp.max()
    rp_norm = (rp / rp_max).clip(0, 1) if rp_max > 0 else pd.Series(0.0, index=gdf.index)

    daily_traffic = rp_norm.map(_estimate_daily_traffic)
    length_km = gdf["length_km"].fillna(0).clip(lower=0)

    # Annual VMT proxy in vehicle-km
    annual_vmt = daily_traffic * length_km * 365
    gdf["annual_vmt_proxy"] = annual_vmt.round(0).astype(int)

    # Fatal crashes averted per year
    nilsson = gdf.get("nilsson_reduction_factor", pd.Series(0.0, index=gdf.index)).fillna(0)
    region_col = gdf.get("region", pd.Series("default", index=gdf.index))

    crash_rates = region_col.map(
        lambda r: CRASH_RATE_PER_100M_VMT.get(str(r).lower(), CRASH_RATE_PER_100M_VMT["default"])
    )
    vosl = region_col.map(
        lambda r: VOSL_USD.get(str(r).lower(), VOSL_USD["default"])
    )

    crashes_averted = (annual_vmt / 1e8) * crash_rates * nilsson
    gdf["crashes_averted_proxy"] = crashes_averted.round(4)

    economic_benefit = crashes_averted * vosl
    gdf["economic_benefit_usd"] = economic_benefit.round(0).astype(int)
    gdf["economic_benefit_m"] = (economic_benefit / 1e6).round(3)

    return gdf
