"""
Load and clean the ADB GeoJSON datasets into analysis-ready GeoDataFrames.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Literal

from src.config import MAHARASHTRA_GEOJSON, THAILAND_GEOJSON


REGION_META = {
    "maharashtra": {
        "path": MAHARASHTRA_GEOJSON,
        "region": "Maharashtra",
        "country": "India",
        "crs_utm": "EPSG:32643",   # UTM zone 43N covers Maharashtra
    },
    "thailand": {
        "path": THAILAND_GEOJSON,
        "region": "Thailand",
        "country": "Thailand",
        "crs_utm": "EPSG:32647",   # UTM zone 47N covers Thailand
    },
}


def load_region(
    region: Literal["maharashtra", "thailand"],
    filter_valid: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load a region's GeoJSON and return a clean GeoDataFrame.

    Args:
        region: 'maharashtra' or 'thailand'
        filter_valid: Drop segments with no TomTom speed data (AnalysisStatus != 'Valid'
                      or SampleSize == 0). Set False to keep all geometry.

    Returns:
        GeoDataFrame in WGS84 (EPSG:4326) with standardised columns.
    """
    meta = REGION_META[region]
    print(f"Loading {meta['region']} from {meta['path'].name} …")

    gdf = gpd.read_file(meta["path"])
    gdf["region"] = meta["region"]
    gdf["country"] = meta["country"]

    # ── Standardise column names ───────────────────────────────────────────
    if "Sample_Size_Total" in gdf.columns:
        gdf = gdf.rename(columns={"Sample_Size_Total": "SampleSizeTotal"})

    # Maharashtra has both 'class' and 'RoadClass'; prefer 'RoadClass', drop 'class'
    if "class" in gdf.columns:
        if "RoadClass" not in gdf.columns:
            gdf = gdf.rename(columns={"class": "RoadClass"})
        else:
            # RoadClass already exists — fill nulls from 'class', then drop 'class'
            gdf["RoadClass"] = gdf["RoadClass"].fillna(gdf["class"])
            gdf = gdf.drop(columns=["class"])

    # ── Coerce numeric columns ─────────────────────────────────────────────
    numeric_cols = [
        "SpeedLimit", "MedianSpeed", "F85thPercentileSpeed",
        "PercentOverLimit", "NumberOverLimit", "WeightedSample",
        "SampleSize_avg", "SampleSizeTotal", "RankedPercentile",
        "Shape_Length",
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # ── Normalise LandUse ──────────────────────────────────────────────────
    if "LandUse" in gdf.columns:
        gdf["LandUse"] = gdf["LandUse"].str.upper().str.strip()

    # ── Normalise RoadClass ────────────────────────────────────────────────
    if "RoadClass" in gdf.columns:
        gdf["RoadClass"] = gdf["RoadClass"].str.lower().str.strip()

    # ── Filter to segments with usable speed data ──────────────────────────
    if filter_valid:
        has_speed = (
            gdf["SpeedLimit"].notna() &
            gdf["MedianSpeed"].notna() &
            (gdf["MedianSpeed"] > 0) &
            (gdf["SampleSize_avg"].fillna(0) > 0)
        )
        if "AnalysisStatus" in gdf.columns:
            has_speed = has_speed & (gdf["AnalysisStatus"] == "Valid")
        n_before = len(gdf)
        gdf = gdf[has_speed].copy()
        print(f"  Kept {len(gdf):,} / {n_before:,} segments with valid speed data")

    # ── Segment length in km ──────────────────────────────────────────────
    gdf["length_km"] = gdf["Shape_Length"] / 1000.0

    # ── Midpoint coordinate for Mapillary queries ─────────────────────────
    gdf["midpoint"] = gdf.geometry.interpolate(0.5, normalized=True)
    gdf["mid_lon"] = gdf["midpoint"].x
    gdf["mid_lat"] = gdf["midpoint"].y
    gdf = gdf.drop(columns=["midpoint"])

    print(f"  Road classes: {gdf['RoadClass'].value_counts().to_dict()}")
    print(f"  Land use: {gdf['LandUse'].value_counts().to_dict()}")

    return gdf


def load_all(filter_valid: bool = True) -> gpd.GeoDataFrame:
    """Load and concatenate both regions into a single GeoDataFrame."""
    mh = load_region("maharashtra", filter_valid=filter_valid)
    th = load_region("thailand", filter_valid=filter_valid)
    gdf = pd.concat([mh, th], ignore_index=True)
    print(f"\nCombined: {len(gdf):,} segments total")
    return gdf


def load_helmet_spi() -> pd.DataFrame:
    """Load helmet wearing SPI data from the Excel file."""
    from src.config import HELMET_SPI_XLSX
    df = pd.read_excel(HELMET_SPI_XLSX, sheet_name=0)
    df.columns = ["Location", "LandUse", "User", "Year", "SPI", "FID"]
    return df
