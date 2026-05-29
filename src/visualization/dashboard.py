"""
Geospatial visualization: interactive Folium map of Speed Safety Scores.

Produces a self-contained HTML file deployable on GitHub Pages.
Features:
  - Choropleth road segments coloured A–E
  - Click segment → popup with score breakdown, Mapillary street image, counterfactual impact
  - Layer controls (by grade, by region)
  - Summary statistics panel
"""
import json
from pathlib import Path
from typing import Optional

import folium
from folium import plugins
import geopandas as gpd
import pandas as pd

from src.config import SCORE_BANDS, MAPS_DIR


GRADE_COLORS = {band[2]: band[4] for band in SCORE_BANDS}  # grade → hex color
GRADE_WEIGHT = {"A": 2, "B": 2, "C": 3, "D": 4, "E": 5}
GRADE_OPACITY = {"A": 0.4, "B": 0.5, "C": 0.7, "D": 0.85, "E": 1.0}


def _score_col(gdf: gpd.GeoDataFrame) -> str:
    return "final_score" if "final_score" in gdf.columns else "speed_safety_score"


def _grade_col(gdf: gpd.GeoDataFrame) -> str:
    return "final_grade" if "final_grade" in gdf.columns else "score_grade"


def _label_col(gdf: gpd.GeoDataFrame) -> str:
    return "final_label" if "final_label" in gdf.columns else "score_label"


def _mapillary_img_html(street_link: Optional[str]) -> str:
    if not street_link or pd.isna(street_link):
        return ""
    parts = [p.strip() for p in str(street_link).split(",")]
    if len(parts) >= 2:
        try:
            lon, lat = float(parts[0]), float(parts[1])
            url = f"https://maps.googleapis.com/maps/api/streetview?size=300x150&location={lat},{lon}&fov=90&source=outdoor"
            return f'<img src="{url}" width="300" style="border-radius:4px;margin-top:6px;" onerror="this.style.display=\'none\'">'
        except ValueError:
            pass
    return ""


def _segment_popup(row: pd.Series, score_col: str, grade_col: str, label_col: str) -> str:
    score = row.get(score_col, "N/A")
    grade = row.get(grade_col, "?")
    label = row.get(label_col, "")
    road_class = row.get("RoadClass", "unknown")
    land_use = row.get("LandUse", "unknown")
    speed_limit = row.get("SpeedLimit", "N/A")
    f85 = row.get("F85thPercentileSpeed", "N/A")
    threshold = row.get("safe_system_threshold_kmh", "N/A")
    excess = row.get("speed_excess_kmh", 0)
    pct_over = row.get("PercentOverLimit", 0)
    impact = row.get("nilsson_reduction_factor", 0)
    region = row.get("region", "")
    name = row.get("english_ro") or row.get("names_primary") or "Unnamed segment"
    color = GRADE_COLORS.get(grade, "#666")
    img_html = _mapillary_img_html(row.get("StreetImageLink"))

    reduction_pct = f"{impact*100:.1f}%" if isinstance(impact, float) else "N/A"

    return f"""
<div style="font-family:sans-serif;min-width:280px;max-width:320px">
  <div style="background:{color};color:white;padding:8px 12px;border-radius:4px 4px 0 0;">
    <b style="font-size:1.1em">Grade {grade}: {label}</b>
    <span style="float:right;font-size:1.4em;font-weight:bold">{score:.1f}/100</span>
  </div>
  <div style="padding:10px 12px;border:1px solid #ddd;border-top:none;border-radius:0 0 4px 4px">
    <p style="margin:2px 0"><b>Road:</b> {name}</p>
    <p style="margin:2px 0"><b>Class:</b> {road_class.title()} | <b>Context:</b> {land_use.title()}</p>
    <hr style="margin:6px 0">
    <p style="margin:2px 0"><b>Posted limit:</b> {speed_limit} km/h</p>
    <p style="margin:2px 0"><b>85th-pct speed:</b> {f85} km/h</p>
    <p style="margin:2px 0"><b>Safe System threshold:</b> {threshold} km/h</p>
    <p style="margin:2px 0"><b>Speed excess above threshold:</b> <span style="color:{color};font-weight:bold">{excess} km/h</span></p>
    <p style="margin:2px 0"><b>Vehicles exceeding limit:</b> {pct_over*100:.1f}%</p>
    <hr style="margin:6px 0">
    <p style="margin:2px 0"><b>Estimated fatality reduction</b> if limit reduced to threshold: <b>{reduction_pct}</b></p>
    <p style="margin:2px 0;font-size:0.85em;color:#666">Region: {region.title()}</p>
    {img_html}
  </div>
</div>"""


def build_map(
    gdf: gpd.GeoDataFrame,
    title: str = "AI for Safer Roads — Speed Safety Score",
    out_path: Optional[Path] = None,
) -> folium.Map:
    """
    Build a full interactive Folium map from a scored GeoDataFrame.

    Args:
        gdf: GeoDataFrame with speed_safety_score (or final_score) and geometry
        title: Map title shown in the legend panel
        out_path: Save HTML to this path (default: outputs/maps/speed_safety_map.html)

    Returns:
        folium.Map object
    """
    score_col = _score_col(gdf)
    grade_col = _grade_col(gdf)
    label_col = _label_col(gdf)

    # Centre map on data centroid
    centroid = gdf.geometry.union_all().centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=6,
        tiles="CartoDB positron",
        attr="© CartoDB © OpenStreetMap contributors",
    )

    # ── Simplify geometries for compact HTML output ────────────────────────
    # Tolerance ~0.001° ≈ 100m — enough detail for a web map, much smaller file
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.001, preserve_topology=True)

    # ── Layer per grade ────────────────────────────────────────────────────
    grade_groups = {}
    for lo, hi, grade, label, color in SCORE_BANDS:
        subset = gdf[gdf[grade_col] == grade].copy()
        if subset.empty:
            continue
        fg = folium.FeatureGroup(
            name=f"Grade {grade} — {label} ({len(subset):,} segments)",
            show=(grade in ("D", "E")),   # only D/E visible by default
        )

        # Build GeoJSON FeatureCollection; add tooltips via style_function
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": {
                        "grade": grade,
                        "score": round(float(row.get(score_col, 0)), 1),
                        "road": row.get("RoadClass", ""),
                        "land_use": row.get("LandUse", ""),
                        "speed_limit": row.get("SpeedLimit"),
                        "f85": row.get("F85thPercentileSpeed"),
                        "threshold": row.get("safe_system_threshold_kmh"),
                        "excess": row.get("speed_excess_kmh", 0),
                        "pct_over": round(float(row.get("PercentOverLimit", 0) or 0) * 100, 1),
                        "nilsson": round(float(row.get("nilsson_reduction_factor", 0) or 0) * 100, 1),
                        "region": row.get("region", ""),
                        "name": str(row.get("english_ro") or row.get("names_primary") or ""),
                        "street_link": str(row.get("StreetImageLink") or ""),
                    },
                }
                for _, row in subset.iterrows()
            ],
        }

        folium.GeoJson(
            geojson_data,
            name=f"Grade {grade}",
            style_function=lambda f, c=color, g=grade: {
                "color": c,
                "weight": GRADE_WEIGHT[g],
                "opacity": GRADE_OPACITY[g],
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["grade", "score", "road", "land_use", "speed_limit", "f85", "threshold"],
                aliases=["Grade", "Score", "Road class", "Land use", "Posted limit (km/h)",
                         "85th-pct speed (km/h)", "Safe System threshold (km/h)"],
                localize=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["grade", "score", "name", "road", "land_use",
                        "speed_limit", "f85", "threshold", "excess",
                        "pct_over", "nilsson", "region"],
                aliases=["Grade", "Score/100", "Road name", "Class", "Land use",
                         "Posted limit", "85th-pct speed", "SS Threshold",
                         "Speed excess (km/h)", "% exceeding limit",
                         "Fatality reduction if reduced (%)", "Region"],
                max_width=380,
            ),
        ).add_to(fg)

        fg.add_to(m)
        grade_groups[grade] = fg

    # ── Legend ────────────────────────────────────────────────────────────
    legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:14px 18px;border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.25);font-family:sans-serif;min-width:220px">
  <b style="font-size:0.95em">{title}</b><br>
  <small style="color:#666">Speed Safety Score (0–100)</small>
  <hr style="margin:8px 0">
"""
    for lo, hi, grade, label, color in SCORE_BANDS:
        count = len(gdf[gdf[grade_col] == grade])
        legend_html += (
            f'<div style="margin:4px 0">'
            f'<span style="display:inline-block;width:20px;height:10px;'
            f'background:{color};border-radius:2px;margin-right:6px"></span>'
            f'<b>Grade {grade}</b> {label} <small style="color:#888">'
            f'({lo}–{hi}) · {count:,}</small></div>\n'
        )
    legend_html += """
  <hr style="margin:8px 0">
  <small style="color:#555">Based on WHO Safe System thresholds.<br>
  Click any segment for full details.</small>
</div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Fullscreen + layer control ─────────────────────────────────────────
    plugins.Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Save ──────────────────────────────────────────────────────────────
    if out_path is None:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MAPS_DIR / "speed_safety_map.html"
    m.save(str(out_path))
    print(f"Map saved → {out_path}")
    return m


def build_region_maps(scored_dir: Path = None):
    """Build individual maps for each region + a combined map."""
    if scored_dir is None:
        scored_dir = MAPS_DIR.parent.parent / "outputs" / "scores"

    for region in ["maharashtra", "thailand"]:
        # Use final scores if available, else stage1
        final = scored_dir / f"{region}_final.geojson"
        stage1 = scored_dir / f"{region}_stage1.geojson"
        path = final if final.exists() else stage1
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        build_map(
            gdf,
            title=f"Speed Safety Score — {region.title()}",
            out_path=MAPS_DIR / f"{region}_map.html",
        )

    # Combined
    combined = scored_dir / "combined_stage1.geojson"
    if combined.exists():
        gdf = gpd.read_file(combined)
        build_map(
            gdf,
            title="Speed Safety Score — Maharashtra & Thailand",
            out_path=MAPS_DIR / "combined_map.html",
        )
