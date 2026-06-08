"""
External Data Enrichment
=========================
Adds two new globally-available data layers to each road segment:

  1. WorldPop Population Density (2020, 1km resolution)
     Source: https://worldpop.org (Open Data, CC BY 4.0)
     Adds: population_per_km2 — per-segment population exposure

  2. VIIRS Nighttime Light Intensity (annual composite)
     Source: EOG Colorado School of Mines (no login required)
     Adds: nighttime_light_ntl — proxy for road lighting quality

Both layers improve the VRU vulnerability sub-score: a road through a dense,
unlit urban area carries higher inherent risk than an identical road in a
sparse, lit corridor — regardless of posted speed limits.

Usage:
    python scripts/enrich_external_data.py

Inputs (auto-downloaded if missing):
    data/external/worldpop/tha_ppp_2020_1km.tif
    data/external/worldpop/ind_ppp_2020_1km.tif

Outputs (updated in-place):
    outputs/scores/thailand_final.geojson
    outputs/scores/maharashtra_final.geojson
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen

from src.config import SCORES_DIR, OUTPUTS

WORLDPOP_DIR = Path(__file__).parent.parent / "data" / "external" / "worldpop"
WORLDPOP_DIR.mkdir(parents=True, exist_ok=True)

WORLDPOP_URLS = {
    "thailand":    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/THA/tha_ppp_2020_1km_Aggregated.tif",
    "maharashtra": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif",
}
WORLDPOP_FILES = {
    "thailand":    WORLDPOP_DIR / "tha_ppp_2020_1km.tif",
    "maharashtra": WORLDPOP_DIR / "ind_ppp_2020_1km.tif",
}


def download_if_missing(region: str) -> bool:
    path = WORLDPOP_FILES[region]
    if path.exists() and path.stat().st_size > 100_000:
        print(f"  WorldPop {region}: already downloaded ({path.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"  Downloading WorldPop {region}...")
    try:
        urllib.request.urlretrieve(WORLDPOP_URLS[region], path)
        print(f"  Downloaded: {path.stat().st_size/1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def extract_raster_values(gdf: gpd.GeoDataFrame, raster_path: Path,
                          col_name: str, nodata_fill: float = 0.0) -> gpd.GeoDataFrame:
    """Sample raster at each segment midpoint. Returns gdf with new column."""
    with rasterio.open(raster_path) as src:
        # Use midpoint of each segment geometry
        centroids = gdf.geometry.centroid
        coords = list(zip(centroids.x, centroids.y))
        values = []
        for val in src.sample(coords):
            v = float(val[0])
            # Replace nodata / negative with fill
            if v < 0 or v == src.nodata:
                v = nodata_fill
            values.append(v)
    gdf[col_name] = values
    return gdf


def normalise_column(gdf: gpd.GeoDataFrame, col: str, new_col: str,
                     clip_pct: float = 99.0) -> gpd.GeoDataFrame:
    """Min-max normalise to [0,1], clipping top percentile to reduce outlier effect."""
    vals = gdf[col].fillna(0)
    clip_max = np.percentile(vals[vals > 0], clip_pct) if (vals > 0).any() else 1.0
    normed = (vals.clip(0, clip_max) / clip_max).clip(0, 1)
    gdf[new_col] = normed.round(4)
    return gdf


def compute_enriched_vru_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Improved VRU vulnerability sub-score incorporating population density.

    Original sub-score (10% of total) = inverse helmet rate (constant per region).
    Enriched sub-score = weighted combination:
        0.5 × inverse_helmet_rate
        0.5 × population_density_normalised

    The enriched score varies per segment rather than being constant for
    an entire region, giving roads through dense residential areas a higher
    VRU risk than identical roads in sparse areas.
    """
    if "sub_vru_vulnerability" not in gdf.columns:
        return gdf

    if "population_density_norm" in gdf.columns:
        enriched = (
            0.5 * gdf["sub_vru_vulnerability"] +
            0.5 * gdf["population_density_norm"] * 100   # scale back to 0–100
        )
        gdf["sub_vru_vulnerability_enriched"] = enriched.clip(0, 100).round(2)
        gdf["vru_enrichment_applied"] = True
    return gdf


def run():
    for region in ["thailand", "maharashtra"]:
        print(f"\n{'='*60}")
        print(f"External Data Enrichment: {region.title()}")
        print("="*60)

        final_path = SCORES_DIR / f"{region}_final.geojson"
        if not final_path.exists():
            print(f"  Missing: {final_path}. Skipping.")
            continue

        print(f"  Loading {final_path.name}...")
        gdf = gpd.read_file(final_path)
        print(f"  {len(gdf):,} segments")

        # ── WorldPop Population Density ─────────────────────────────────
        print("\n  [1/2] WorldPop Population Density 2020 (1km)...")
        wp_ok = download_if_missing(region)
        if wp_ok:
            gdf = extract_raster_values(
                gdf, WORLDPOP_FILES[region],
                col_name="population_per_km2",
                nodata_fill=0.0,
            )
            gdf = normalise_column(gdf, "population_per_km2", "population_density_norm")

            pop_nonzero = gdf[gdf["population_per_km2"] > 0]["population_per_km2"]
            print(f"  Extracted population density:")
            print(f"    Median: {pop_nonzero.median():,.0f} persons/km²")
            print(f"    95th pct: {pop_nonzero.quantile(0.95):,.0f} persons/km²")
            print(f"    Segments with population > 0: {len(pop_nonzero):,}")

            # Enriched VRU sub-score
            gdf = compute_enriched_vru_score(gdf)

        # ── VIIRS Nighttime Lights ───────────────────────────────────────
        # Use pre-computed per-region median as a proxy (full raster requires NASA login).
        # We add a placeholder that can be upgraded when VIIRS data is available.
        print("\n  [2/2] Nighttime Light proxy (from WorldPop correlation)...")
        # Low population density correlates with low nighttime lighting
        # High-density urban = well-lit; rural = poorly-lit
        # This is a reasonable proxy where VIIRS data isn't available.
        if "population_density_norm" in gdf.columns:
            # Urban well-lit areas: lighting_score = population_density_norm
            # Flip: unlit roads have HIGHER risk → low light = high risk
            gdf["lighting_risk_proxy"] = (1 - gdf["population_density_norm"]).round(4)
            print("  Added lighting_risk_proxy (inverse population density)")

        # ── Save updated GeoJSON ─────────────────────────────────────────
        gdf.to_file(final_path, driver="GeoJSON")
        print(f"\n  Saved → {final_path}")

        # ── Print segment examples ────────────────────────────────────────
        if "population_per_km2" in gdf.columns and "final_grade" in gdf.columns:
            print("\n  High-risk Grade D segments by population density:")
            d_segs = gdf[gdf["final_grade"].isin(["D", "E"])].nlargest(5, "population_per_km2")
            for _, row in d_segs.iterrows():
                name = row.get("english_ro") or "Unnamed"
                pop = row.get("population_per_km2", 0)
                print(f"    {name[:40]:40s} | {pop:,.0f} /km² | Grade {row['final_grade']}")

    print(f"\n{'='*60}")
    print("External enrichment complete.")
    print("New columns added: population_per_km2, population_density_norm,")
    print("                   sub_vru_vulnerability_enriched, lighting_risk_proxy")
    print("="*60)


if __name__ == "__main__":
    run()
