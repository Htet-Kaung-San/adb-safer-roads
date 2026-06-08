"""
Unit tests for the Safe System scoring logic.
Run: python -m pytest tests/test_scoring.py -v
"""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.safe_system import (
    get_safe_system_threshold,
    compute_speed_safety_score,
)
from src.config import (
    SAFE_SYSTEM_THRESHOLDS,
    SCORE_BANDS,
    VOSL_USD,
)


def make_segment(**kwargs):
    """Helper: build a minimal one-row GeoDataFrame."""
    defaults = {
        "OBJECTID": 1,
        "F85thPercentileSpeed": 80.0,
        "SpeedLimit": 80.0,
        "PercentOverLimit": 20.0,
        "RankedPercentile": 0.5,
        "LandUse": "URBAN",
        "RoadClass": "primary",
        "sub_vru_vulnerability": 0.5,
        "Shape_Length": 500.0,
        "geometry": LineString([(100, 13), (100.005, 13)]),
    }
    defaults.update(kwargs)
    return gpd.GeoDataFrame([defaults], crs="EPSG:4326")


class TestSafeSystemThresholds:
    """Thresholds must match the SAFE_SYSTEM_THRESHOLDS config (upper-case land use keys)."""

    def test_urban_motorway(self):
        assert get_safe_system_threshold("motorway", "URBAN") == 80

    def test_rural_motorway(self):
        assert get_safe_system_threshold("motorway", "RURAL") == 110

    def test_urban_primary(self):
        assert get_safe_system_threshold("primary", "URBAN") == 50

    def test_rural_secondary(self):
        assert get_safe_system_threshold("secondary", "RURAL") == 60

    def test_urban_trunk(self):
        assert get_safe_system_threshold("trunk", "URBAN") == 60

    def test_urban_secondary(self):
        assert get_safe_system_threshold("secondary", "URBAN") == 40

    def test_unknown_class_urban_fallback(self):
        val = get_safe_system_threshold("unknown_class", "URBAN")
        # Should fall back to (None, "URBAN") = 50
        assert val == 50

    def test_known_class_unknown_land_fallback(self):
        val = get_safe_system_threshold("primary", None)
        # Should fall back to ("primary", None) = 60
        assert val == 60

    def test_all_thresholds_positive(self):
        for (cls, lu), threshold in SAFE_SYSTEM_THRESHOLDS.items():
            assert threshold > 0, f"Threshold for ({cls}, {lu}) must be positive"

    def test_all_thresholds_reasonable_speed_range(self):
        for (cls, lu), threshold in SAFE_SYSTEM_THRESHOLDS.items():
            assert 20 <= threshold <= 130, f"Threshold {threshold} for ({cls}, {lu}) is unreasonable"


class TestSpeedSafetyScore:
    def test_safe_segment_gets_low_score(self):
        """A segment well within Safe System limits should score < 40 (Grade A or B)."""
        gdf = make_segment(
            F85thPercentileSpeed=38.0,  # well below urban primary 50km/h threshold
            SpeedLimit=50.0,
            PercentOverLimit=3.0,
            RankedPercentile=0.15,
            LandUse="URBAN",
            RoadClass="primary",
            sub_vru_vulnerability=0.05,
        )
        result = compute_speed_safety_score(gdf)
        score = result["speed_safety_score"].iloc[0]
        grade = result["score_grade"].iloc[0]
        assert score < 40, f"Safe segment should score < 40, got {score:.1f}"
        assert grade in ("A", "B"), f"Safe segment should be A or B, got {grade}"

    def test_dangerous_segment_gets_high_score(self):
        """A segment far exceeding Safe System thresholds should score > 60 (Grade D+)."""
        gdf = make_segment(
            F85thPercentileSpeed=115.0,  # 115 vs 50km/h urban primary threshold
            SpeedLimit=90.0,
            PercentOverLimit=70.0,
            RankedPercentile=0.95,
            LandUse="URBAN",
            RoadClass="primary",
            sub_vru_vulnerability=0.9,
        )
        result = compute_speed_safety_score(gdf)
        score = result["speed_safety_score"].iloc[0]
        grade = result["score_grade"].iloc[0]
        assert score > 60, f"Dangerous segment should score > 60, got {score:.1f}"
        assert grade in ("D", "E"), f"Dangerous segment should be D or E, got {grade}"

    def test_score_bounded_0_100(self):
        """Score must always be in [0, 100]."""
        extremes = [
            make_segment(F85thPercentileSpeed=1.0, SpeedLimit=10.0, PercentOverLimit=0.0, RankedPercentile=0.01),
            make_segment(F85thPercentileSpeed=200.0, SpeedLimit=200.0, PercentOverLimit=99.0, RankedPercentile=1.0),
        ]
        for gdf in extremes:
            result = compute_speed_safety_score(gdf)
            score = result["speed_safety_score"].iloc[0]
            assert 0 <= score <= 100, f"Score {score} out of [0, 100]"

    def test_grade_matches_score_band(self):
        """Grade must match the SCORE_BANDS config for the given score."""
        gdf = make_segment(
            F85thPercentileSpeed=115.0,
            SpeedLimit=90.0,
            PercentOverLimit=65.0,
            RankedPercentile=0.95,
            sub_vru_vulnerability=0.9,
        )
        result = compute_speed_safety_score(gdf)
        score = result["speed_safety_score"].iloc[0]
        grade = result["score_grade"].iloc[0]
        # Derive expected grade from SCORE_BANDS
        expected = "E"
        for lo, hi, g, lbl, _ in SCORE_BANDS:
            if lo <= score < hi:
                expected = g
                break
        assert grade == expected, f"Grade {grade} doesn't match band-derived {expected} for score {score:.1f}"

    def test_speed_excess_non_negative(self):
        """speed_excess_kmh must be clipped to 0 (can't be negative)."""
        gdf = make_segment(
            F85thPercentileSpeed=30.0,  # well below any threshold
            SpeedLimit=50.0,
            LandUse="URBAN",
            RoadClass="primary",
        )
        result = compute_speed_safety_score(gdf)
        excess = result["speed_excess_kmh"].iloc[0]
        assert excess >= 0, f"speed_excess_kmh should be >= 0, got {excess}"

    def test_speed_excess_correct_value(self):
        """speed_excess_kmh = max(0, F85th - threshold)."""
        gdf = make_segment(
            F85thPercentileSpeed=95.0,  # threshold for urban primary = 50km/h
            SpeedLimit=80.0,
            LandUse="URBAN",
            RoadClass="primary",
        )
        result = compute_speed_safety_score(gdf)
        excess = result["speed_excess_kmh"].iloc[0]
        threshold = result["safe_system_threshold_kmh"].iloc[0]
        expected = max(0, 95.0 - threshold)
        assert abs(excess - expected) < 1.0, f"Expected speed_excess ~{expected:.1f}, got {excess:.1f}"

    def test_monotone_score_increase_with_operating_speed(self):
        """Holding everything else equal, higher operating speed → higher or equal score."""
        scores = []
        for speed in [50, 70, 90, 110, 130]:
            gdf = make_segment(
                F85thPercentileSpeed=float(speed),
                SpeedLimit=80.0,
                LandUse="URBAN",
                RoadClass="primary",
            )
            result = compute_speed_safety_score(gdf)
            scores.append(result["speed_safety_score"].iloc[0])
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-6, (
                f"Score not monotone: {scores[i]:.1f} at speed idx {i} > {scores[i+1]:.1f} at {i+1}"
            )

    def test_score_columns_present(self):
        """All expected output columns must be present."""
        gdf = make_segment()
        result = compute_speed_safety_score(gdf)
        required = [
            "speed_safety_score", "score_grade", "score_label",
            "safe_system_threshold_kmh", "speed_excess_kmh",
            "sub_speed_deviation", "sub_posted_limit_excess",
            "sub_speeding_prevalence", "sub_traffic_exposure",
        ]
        for col in required:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_score_bands_partition_0_to_100(self):
        """SCORE_BANDS must cover [0, 100) with no gaps or overlaps."""
        sorted_bands = sorted(SCORE_BANDS, key=lambda b: b[0])
        assert sorted_bands[0][0] == 0, "First band must start at 0"
        for i in range(len(sorted_bands) - 1):
            assert sorted_bands[i][1] == sorted_bands[i + 1][0], (
                f"Gap between band {sorted_bands[i][2]} and {sorted_bands[i+1][2]}"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
