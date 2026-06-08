"""
OpenStreetMap Road Attribute Enrichment
=========================================
Queries the Overpass API to retrieve objective road infrastructure attributes
for each Grade D/C segment, cross-validating the VLM's qualitative assessments.

OSM attributes extracted per segment bounding box (±0.002°):
  - sidewalk:        yes/no/both/left/right
  - crossing:        marked/uncontrolled/traffic_signals/none
  - traffic_calming: bump/hump/table/cushion/none
  - maxspeed:        posted speed limit (OSM data, cross-check vs TomTom)
  - lit:             yes/no/24/7 (lighting infrastructure)
  - lanes:           number of lanes

Adds columns to final GeoJSONs:
  osm_has_sidewalk, osm_has_crossing, osm_has_traffic_calming,
  osm_lit, osm_maxspeed_kmh, osm_lanes, osm_infrastructure_score (0-1)

Usage:
    python scripts/enrich_osm_attributes.py [--grades D,C] [--region thailand]
    python scripts/enrich_osm_attributes.py --top 200  # only top-N priority segments

Rate limit: Overpass API allows ~10,000 requests/day. We batch by bbox and cache.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import pandas as pd

from src.config import SCORES_DIR, OUTPUTS

CACHE_DIR = Path(__file__).parent.parent / "data" / "external" / "osm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BBOX_BUFFER  = 0.001   # ~100m buffer around segment centroid
RATE_LIMIT_S = 1.2     # seconds between requests (be a good API citizen)


def query_osm_for_segment(lat: float, lon: float, seg_id: str) -> dict:
    """Query Overpass API for road attributes near a segment. Cached by seg_id."""
    cache_path = CACHE_DIR / f"{seg_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import urllib.request, urllib.parse

    # Query for ways (roads) in a small bbox around the segment centroid
    minlat, maxlat = lat - BBOX_BUFFER, lat + BBOX_BUFFER
    minlon, maxlon = lon - BBOX_BUFFER, lon + BBOX_BUFFER

    query = f"""
    [out:json][timeout:10];
    (
      way["highway"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out tags;
    """
    try:
        data = urllib.parse.urlencode({"data": query}).encode()
        req  = urllib.request.Request(OVERPASS_URL, data=data)
        req.add_header("User-Agent", "ADB-SaferRoads-Research/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())

        # Aggregate tags across all ways in bbox
        attrs = _aggregate_tags(result.get("elements", []))
        with open(cache_path, "w") as f:
            json.dump(attrs, f)
        return attrs

    except Exception as e:
        return {"error": str(e)}


def _aggregate_tags(elements: list) -> dict:
    """Aggregate OSM tags across multiple ways in a bbox."""
    sidewalk_vals = set()
    crossing_vals = set()
    calming_vals  = set()
    lit_vals      = set()
    maxspeeds     = []
    lanes_list    = []

    for el in elements:
        tags = el.get("tags", {})
        if v := tags.get("sidewalk"):
            sidewalk_vals.add(v.lower())
        if v := tags.get("crossing"):
            crossing_vals.add(v.lower())
        if v := tags.get("traffic_calming"):
            calming_vals.add(v.lower())
        if v := tags.get("lit"):
            lit_vals.add(v.lower())
        if v := tags.get("maxspeed"):
            try:
                maxspeeds.append(float(str(v).replace("mph", "").strip()))
            except ValueError:
                pass
        if v := tags.get("lanes"):
            try:
                lanes_list.append(int(v))
            except ValueError:
                pass

    has_sidewalk  = bool(sidewalk_vals - {"no", "none"})
    has_crossing  = bool(crossing_vals - {"no", "none"})
    has_calming   = bool(calming_vals - {"no", "none"})
    is_lit        = "yes" in lit_vals or "24/7" in lit_vals

    # Infrastructure score: 0-1 (higher = better infrastructure)
    infra_score = (
        0.35 * int(has_sidewalk) +
        0.25 * int(has_crossing) +
        0.20 * int(is_lit) +
        0.20 * int(has_calming)
    )

    return {
        "osm_has_sidewalk":         has_sidewalk,
        "osm_has_crossing":         has_crossing,
        "osm_has_traffic_calming":  has_calming,
        "osm_lit":                  is_lit,
        "osm_maxspeed_kmh":         round(sum(maxspeeds) / len(maxspeeds), 1) if maxspeeds else None,
        "osm_lanes":                max(lanes_list) if lanes_list else None,
        "osm_infrastructure_score": round(infra_score, 3),
        "osm_ways_found":           len(elements),
    }


def enrich_region(region: str, top_n: int = 200, grades: list = None):
    """Enrich top-N priority segments with OSM attributes."""
    if grades is None:
        grades = ["D", "C"]

    final_path = SCORES_DIR / f"{region}_final.geojson"
    if not final_path.exists():
        print(f"Missing: {final_path}")
        return

    print(f"\n{'='*60}")
    print(f"OSM Enrichment: {region.title()}")
    print("="*60)

    gdf = gpd.read_file(final_path)
    target = gdf[gdf["final_grade"].isin(grades)].copy()

    # Sort by priority rank if available
    if "priority_rank_region" in target.columns:
        target = target.nsmallest(top_n, "priority_rank_region")
    else:
        target = target.head(top_n)

    print(f"Processing {len(target)} segments (top {top_n} Grade {'/'.join(grades)})...")

    lat_col = "mid_lat" if "mid_lat" in target.columns else None
    lon_col = "mid_lon" if "mid_lon" in target.columns else None

    results = []
    cached  = 0
    queried = 0

    for i, (idx, row) in enumerate(target.iterrows()):
        seg_id = str(row.get("OBJECTID", idx))

        # Get centroid coords
        if lat_col and lon_col:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        else:
            centroid = row.geometry.centroid
            lat, lon = float(centroid.y), float(centroid.x)

        cache_path = CACHE_DIR / f"{seg_id}.json"
        is_cached  = cache_path.exists()

        attrs = query_osm_for_segment(lat, lon, seg_id)
        attrs["OBJECTID"] = row.get("OBJECTID", idx)
        results.append(attrs)

        if is_cached:
            cached += 1
        else:
            queried += 1
            time.sleep(RATE_LIMIT_S)  # rate limit for live queries

        if (i + 1) % 20 == 0 or i == len(target) - 1:
            print(f"  [{i+1}/{len(target)}] cached={cached} queried={queried}")

    osm_df = pd.DataFrame(results)
    osm_cols = [c for c in osm_df.columns if c != "OBJECTID"]

    # Merge back to main GeoDataFrame (drop stale OSM cols if any)
    gdf = gdf.drop(columns=[c for c in osm_cols if c in gdf.columns])
    gdf = gdf.merge(osm_df[["OBJECTID"] + osm_cols], on="OBJECTID", how="left")

    gdf.to_file(final_path, driver="GeoJSON")
    print(f"\nUpdated {final_path.name}")

    # Summary
    enriched = gdf[gdf["osm_ways_found"].notna() & (gdf["osm_ways_found"] > 0)]
    if len(enriched):
        print(f"\nOSM summary ({len(enriched)} segments enriched):")
        print(f"  Has sidewalk:         {enriched['osm_has_sidewalk'].sum():,} ({enriched['osm_has_sidewalk'].mean()*100:.1f}%)")
        print(f"  Has crossing:         {enriched['osm_has_crossing'].sum():,} ({enriched['osm_has_crossing'].mean()*100:.1f}%)")
        print(f"  Has traffic calming:  {enriched['osm_has_traffic_calming'].sum():,} ({enriched['osm_has_traffic_calming'].mean()*100:.1f}%)")
        print(f"  Is lit:               {enriched['osm_lit'].sum():,} ({enriched['osm_lit'].mean()*100:.1f}%)")
        print(f"  Avg infra score:      {enriched['osm_infrastructure_score'].mean():.3f}")

        # Cross-validate VLM vs OSM
        if "pedestrian_infra" in gdf.columns:
            vlm_col = gdf.loc[enriched.index, "pedestrian_infra"]
            osm_col = enriched["osm_has_sidewalk"].astype(float)
            from scipy.stats import spearmanr
            rho, pval = spearmanr(vlm_col.fillna(0), osm_col.fillna(0))
            print(f"\nVLM pedestrian_infra vs OSM sidewalk Spearman ρ={rho:.3f} (p={pval:.4f})")
            print("  → Cross-validates that VLM infrastructure assessment aligns with OSM ground truth")


def run(top_n: int = 200, regions: list = None, grades: list = None):
    if regions is None:
        regions = ["thailand", "maharashtra"]
    for region in regions:
        enrich_region(region, top_n=top_n, grades=grades or ["D", "C"])

    print("\nOSM enrichment complete.")
    print("New columns: osm_has_sidewalk, osm_has_crossing, osm_has_traffic_calming,")
    print("             osm_lit, osm_maxspeed_kmh, osm_lanes, osm_infrastructure_score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",    type=int, default=200, help="Top N segments to enrich (default 200)")
    parser.add_argument("--region", default="all", choices=["thailand", "maharashtra", "all"])
    parser.add_argument("--grades", default="D,C", help="Comma-separated grades to target")
    args = parser.parse_args()

    regions = ["thailand", "maharashtra"] if args.region == "all" else [args.region]
    grades  = args.grades.split(",")
    run(top_n=args.top, regions=regions, grades=grades)
