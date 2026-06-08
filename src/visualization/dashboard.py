"""
Geospatial visualization: interactive Folium map of Speed Safety Scores.

Produces a self-contained HTML file deployable on GitHub Pages.

Features (upgraded):
  - Choropleth road segments coloured A-E
  - Rich popup per segment:
      · Speed profile (posted / 85th-pct / SS threshold / excess)
      · Confidence interval badge (MC Dropout 95% CI)
      · Economic benefit (annual $ value of intervention)
      · VLM visual feature breakdown (bar charts)
      · YOLO pedestrian/VRU counts (if available)
      · Archetype tag with icon + recommended intervention
      · Priority rank badge
  - Summary statistics panel (top-right)
  - Layer controls (by grade, by region)
  - Fullscreen mode
"""
import json
from pathlib import Path
from typing import Optional

import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
import numpy as np

from src.config import SCORE_BANDS, MAPS_DIR


GRADE_COLORS = {band[2]: band[4] for band in SCORE_BANDS}
GRADE_WEIGHT = {"A": 2, "B": 2, "C": 3, "D": 4, "E": 5}
GRADE_OPACITY = {"A": 0.4, "B": 0.5, "C": 0.7, "D": 0.9, "E": 1.0}

VLM_FEATURE_LABELS = {
    "pedestrian_infra":   "Pedestrian infra",
    "cyclist_infra":      "Cyclist infra",
    "roadside_activity":  "Roadside activity",
    "road_condition":     "Road condition",
    "signage_quality":    "Signage quality",
    "vru_exposure":       "VRU exposure",
    "visibility_quality": "Visibility",
}


def _score_col(gdf):
    return "final_score" if "final_score" in gdf.columns else "speed_safety_score"

def _grade_col(gdf):
    return "final_grade" if "final_grade" in gdf.columns else "score_grade"

def _label_col(gdf):
    return "final_label" if "final_label" in gdf.columns else "score_label"


def _fmt_currency(val) -> str:
    try:
        v = float(val)
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        elif v >= 1e3:
            return f"${v/1e3:.0f}K"
        else:
            return f"${v:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def _vlm_bar_html(label: str, value: float, danger_high: bool = True) -> str:
    """Render a single VLM feature as a compact coloured bar."""
    try:
        pct = max(0.0, min(1.0, float(value))) * 100
    except (TypeError, ValueError):
        return ""
    # Colour: for most features, high = bad (red). For road_condition/signage, high = good (green).
    good_features = {"road_condition", "signage_quality", "visibility_quality"}
    feat_key = label.lower().replace(" ", "_")
    if feat_key in good_features or not danger_high:
        bar_color = f"hsl({int(120 * pct / 100)},70%,45%)"  # green when high
    else:
        bar_color = f"hsl({int(120 * (1 - pct / 100))},70%,45%)"  # red when high

    return f"""
<div style="display:flex;align-items:center;margin:2px 0;font-size:0.78em">
  <span style="width:110px;color:#555;flex-shrink:0">{label}</span>
  <div style="flex:1;background:#eee;border-radius:3px;height:8px;margin:0 6px">
    <div style="width:{pct:.0f}%;background:{bar_color};border-radius:3px;height:100%"></div>
  </div>
  <span style="width:32px;text-align:right;color:#333">{pct:.0f}%</span>
</div>"""


def _confidence_badge(score: float, ci_low, ci_high, score_std, uncertain: bool) -> str:
    """Render score with 95% CI and uncertainty flag."""
    try:
        std = float(score_std)
        lo = float(ci_low)
        hi = float(ci_high)
        if np.isnan(std) or np.isnan(lo):
            return f"<b style='font-size:1.3em'>{score:.1f}</b><small style='color:#888'>/100</small>"
        ci_str = f"{lo:.1f}–{hi:.1f}"
        flag = (
            " <span title='Grade may be ambiguous — 95% CI crosses a grade boundary' "
            "style='color:#f39c12;font-size:0.8em'>⚠ uncertain grade</span>"
            if uncertain else ""
        )
        return (
            f"<b style='font-size:1.3em'>{score:.1f}</b>"
            f"<span style='color:#888;font-size:0.8em'> ±{std:.1f}</span>"
            f"<small style='color:#aaa'>/100</small>"
            f"<div style='font-size:0.75em;color:#888'>95% CI: {ci_str}</div>"
            f"{flag}"
        )
    except (TypeError, ValueError):
        return f"<b style='font-size:1.3em'>{score:.1f}</b><small style='color:#888'>/100</small>"


def _priority_badge(rank) -> str:
    try:
        r = int(rank)
        color = "#c0392b" if r <= 10 else "#e67e22" if r <= 50 else "#f1c40f"
        return (
            f"<span style='background:{color};color:white;padding:1px 7px;"
            f"border-radius:10px;font-size:0.8em;font-weight:bold'>"
            f"#{r} priority</span>"
        )
    except (TypeError, ValueError):
        return ""


def _segment_popup(row: pd.Series, score_col: str, grade_col: str, label_col: str) -> str:
    score = row.get(score_col, 0) or 0
    grade = row.get(grade_col, "?")
    label = row.get(label_col, "")
    road_class = row.get("RoadClass", "unknown") or "unknown"
    land_use = row.get("LandUse", "unknown") or "unknown"
    speed_limit = row.get("SpeedLimit", "N/A")
    f85 = row.get("F85thPercentileSpeed", "N/A")
    threshold = row.get("safe_system_threshold_kmh", "N/A")
    excess = row.get("speed_excess_kmh", 0) or 0
    pct_over = row.get("PercentOverLimit", 0) or 0
    nilsson = row.get("nilsson_reduction_factor", 0) or 0
    region = row.get("region", "") or ""
    name = row.get("english_ro") or row.get("names_primary") or "Unnamed segment"
    color = GRADE_COLORS.get(grade, "#666")

    # Economic
    eco_usd = row.get("economic_benefit_usd", None)
    eco_html = ""
    if eco_usd is not None:
        try:
            eco_html = (
                f"<p style='margin:2px 0'><b>Economic value of fix:</b> "
                f"<span style='color:#27ae60;font-weight:bold'>{_fmt_currency(eco_usd)}/year</span></p>"
            )
        except (TypeError, ValueError):
            pass

    # Priority rank
    rank_html = ""
    rank = row.get("priority_rank_region", None)
    if rank and not (isinstance(rank, float) and np.isnan(rank)):
        rank_html = f"<div style='margin:4px 0'>{_priority_badge(rank)}</div>"

    # Confidence interval
    ci_html = _confidence_badge(
        score,
        row.get("score_ci_low"),
        row.get("score_ci_high"),
        row.get("score_std"),
        bool(row.get("grade_uncertain", False)),
    )

    # VLM features section
    vlm_html = ""
    vlm_rows = []
    for feat_key, feat_label in VLM_FEATURE_LABELS.items():
        val = row.get(feat_key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            vlm_rows.append(_vlm_bar_html(feat_label, val))
    if vlm_rows:
        vlm_html = f"""
<div style='margin-top:8px'>
  <p style='margin:0 0 4px 0;font-size:0.85em;font-weight:bold;color:#555'>
    Visual Analysis (VLM Qwen2-72B)</p>
  {''.join(vlm_rows)}
</div>"""

    # YOLO object detection section
    yolo_html = ""
    yolo_ped = row.get("yolo_pedestrian_count")
    yolo_moto = row.get("yolo_moto_count")
    yolo_vru = row.get("yolo_vru_ratio")
    if yolo_vru is not None and not (isinstance(yolo_vru, float) and np.isnan(yolo_vru)):
        try:
            vru_pct = float(yolo_vru) * 100
            yolo_html = f"""
<div style='margin-top:6px;padding:6px;background:#f8f9fa;border-radius:4px'>
  <p style='margin:0 0 2px 0;font-size:0.8em;font-weight:bold;color:#555'>
    Object Detection (YOLOv8)</p>
  <p style='margin:1px 0;font-size:0.78em'>
    Pedestrians: {int(yolo_ped or 0)} &nbsp;|&nbsp; Motorcycles: {int(yolo_moto or 0)} &nbsp;|&nbsp;
    VRU ratio: <b>{vru_pct:.0f}%</b> of detected objects</p>
</div>"""
        except (TypeError, ValueError):
            pass

    # Archetype section
    arch_html = ""
    arch_name = row.get("archetype_name", "Not Applicable")
    if arch_name and arch_name != "Not Applicable":
        arch_color = row.get("archetype_color", "#666") or "#666"
        arch_icon = row.get("archetype_icon", "") or ""
        arch_desc = row.get("archetype_description", "") or ""
        arch_intv = row.get("archetype_intervention", "") or ""
        arch_sec = row.get("archetype_secondary", "") or ""
        arch_html = f"""
<div style='margin-top:8px;padding:8px;background:#f0f4ff;border-radius:4px;
            border-left:3px solid {arch_color}'>
  <p style='margin:0 0 3px 0;font-size:0.85em;font-weight:bold;color:{arch_color}'>
    {arch_icon} Risk Archetype: {arch_name}</p>
  <p style='margin:1px 0;font-size:0.78em;color:#444'>{arch_desc}</p>
  <p style='margin:4px 0 1px 0;font-size:0.78em;font-weight:bold;color:#333'>
    Primary: {arch_intv}</p>
  {'<p style="margin:1px 0;font-size:0.78em;color:#666">Secondary: ' + arch_sec + '</p>' if arch_sec else ''}
</div>"""

    return f"""
<div style="font-family:sans-serif;min-width:300px;max-width:360px;font-size:0.9em">
  <div style="background:{color};color:white;padding:10px 14px;border-radius:6px 6px 0 0">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <b style="font-size:1.05em">Grade {grade}: {label}</b>
        <div style="font-size:0.8em;opacity:0.85">{region.title()} · {road_class.title()} · {land_use.title()}</div>
      </div>
      <div style="text-align:right">{ci_html}</div>
    </div>
    {rank_html}
  </div>
  <div style="padding:10px 14px;border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px">
    <p style="margin:2px 0;font-weight:bold;color:#333">{name}</p>

    <div style="background:#fff8f0;padding:6px 8px;border-radius:4px;margin:6px 0">
      <p style="margin:1px 0"><b>Posted limit:</b> {speed_limit} km/h &nbsp;
         <b>85th-pct:</b> {f85} km/h</p>
      <p style="margin:1px 0"><b>Safe System threshold:</b> {threshold} km/h &nbsp;
         <b>Excess:</b> <span style="color:{color};font-weight:bold">{excess} km/h</span></p>
      <p style="margin:1px 0"><b>Vehicles exceeding limit:</b> {pct_over*100:.1f}%</p>
    </div>

    <div style="background:#f0fff4;padding:6px 8px;border-radius:4px;margin:6px 0">
      <p style="margin:1px 0"><b>Fatality reduction if corrected:</b>
         <span style="color:#27ae60;font-weight:bold">{nilsson*100:.1f}%</span></p>
      {eco_html}
    </div>

    {vlm_html}
    {yolo_html}
    {arch_html}
  </div>
</div>"""


def _summary_panel_html(gdf: gpd.GeoDataFrame, title: str) -> str:
    """Right-side summary statistics panel."""
    grade_col = _grade_col(gdf)
    score_col = _score_col(gdf)

    counts = gdf[grade_col].value_counts()
    total = len(gdf)

    de_segs = gdf[gdf[grade_col].isin(["D", "E"])]
    eco_col = "economic_benefit_usd"
    total_eco = de_segs[eco_col].sum() / 1e6 if eco_col in de_segs.columns else 0
    top_nilsson = 0
    if "nilsson_reduction_factor" in de_segs.columns and len(de_segs):
        top_nilsson = de_segs["nilsson_reduction_factor"].max() * 100

    rows_html = ""
    for lo, hi, grade, label, color in SCORE_BANDS:
        count = counts.get(grade, 0)
        pct = count / total * 100 if total > 0 else 0
        rows_html += f"""
<div style="display:flex;align-items:center;margin:4px 0">
  <span style="width:12px;height:12px;background:{color};border-radius:2px;
               display:inline-block;margin-right:6px;flex-shrink:0"></span>
  <span style="flex:1;font-size:0.85em"><b>Grade {grade}</b> {label}</span>
  <span style="font-size:0.85em;color:#555">{count:,}</span>
  <span style="font-size:0.75em;color:#aaa;margin-left:4px">({pct:.0f}%)</span>
</div>"""

    eco_line = f"""
<div style="background:#e8f8f0;padding:6px 8px;border-radius:4px;margin-top:8px">
  <p style="margin:0;font-size:0.82em;color:#1a7a4a">
    <b>Annual eco. value of Grade D/E fixes:</b><br>${total_eco:.1f}M/year</p>
  <p style="margin:4px 0 0 0;font-size:0.82em;color:#1a7a4a">
    <b>Max fatality reduction (top segment):</b> {top_nilsson:.0f}%</p>
</div>""" if total_eco > 0 else ""

    return f"""
<div style="position:fixed;top:80px;right:15px;z-index:1000;
            background:white;padding:14px 16px;border-radius:8px;
            box-shadow:0 2px 12px rgba(0,0,0,0.2);font-family:sans-serif;
            min-width:220px;max-width:260px">
  <b style="font-size:0.9em;color:#222">{title}</b>
  <div style="font-size:0.78em;color:#888;margin-bottom:8px">
    {total:,} segments analysed
  </div>
  <hr style="margin:0 0 8px 0;border:none;border-top:1px solid #eee">
  {rows_html}
  {eco_line}
  <hr style="margin:8px 0;border:none;border-top:1px solid #eee">
  <p style="margin:0;font-size:0.72em;color:#aaa">
    WHO Safe System thresholds.<br>
    Click any segment for details.<br>
    Grade D/E visible by default.
  </p>
</div>"""


def build_map(
    gdf: gpd.GeoDataFrame,
    title: str = "AI for Safer Roads — Speed Safety Score",
    out_path: Optional[Path] = None,
) -> folium.Map:
    """
    Build a full interactive Folium map from a scored GeoDataFrame.

    Args:
        gdf: GeoDataFrame with final_score (or speed_safety_score) + geometry
        title: Map title shown in the summary panel
        out_path: Save HTML to this path

    Returns:
        folium.Map object
    """
    score_col = _score_col(gdf)
    grade_col = _grade_col(gdf)
    label_col = _label_col(gdf)

    centroid = gdf.geometry.union_all().centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=7,
        tiles="CartoDB positron",
        attr="© CartoDB © OpenStreetMap contributors",
    )

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)

    # ── Layer per grade ────────────────────────────────────────────────────
    for lo, hi, grade, label, color in SCORE_BANDS:
        subset = gdf[gdf[grade_col] == grade].copy()
        if subset.empty:
            continue

        fg = folium.FeatureGroup(
            name=f"Grade {grade} — {label} ({len(subset):,})",
            show=(grade in ("D", "E")),
        )

        for _, row in subset.iterrows():
            popup_html = _segment_popup(row, score_col, grade_col, label_col)
            tooltip_text = (
                f"Grade {row.get(grade_col,'?')} | "
                f"Score {row.get(score_col,0):.1f} | "
                f"{row.get('RoadClass','?')} {row.get('LandUse','?')} | "
                f"85th-pct: {row.get('F85thPercentileSpeed','?')} km/h"
            )

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda f, c=color, g=grade: {
                    "color": c,
                    "weight": GRADE_WEIGHT[g],
                    "opacity": GRADE_OPACITY[g],
                    "fillOpacity": 0,
                },
                tooltip=folium.Tooltip(tooltip_text),
                popup=folium.Popup(popup_html, max_width=380),
            ).add_to(fg)

        fg.add_to(m)

    # ── Summary panel ──────────────────────────────────────────────────────
    m.get_root().html.add_child(
        folium.Element(_summary_panel_html(gdf, title))
    )

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

    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    for region in ["maharashtra", "thailand"]:
        final = scored_dir / f"{region}_final.geojson"
        stage1 = scored_dir / f"{region}_stage1.geojson"
        path = final if final.exists() else stage1
        if not path.exists():
            print(f"  Skipping {region} — no scored GeoJSON found")
            continue
        gdf = gpd.read_file(path)
        build_map(
            gdf,
            title=f"Speed Safety Score — {region.title()}",
            out_path=MAPS_DIR / f"{region}_map.html",
        )

    # Combined (prefer combined_final, fall back to stage1)
    for fname in ["combined_final.geojson", "combined_stage1.geojson"]:
        combined_path = scored_dir / fname
        if combined_path.exists():
            gdf = gpd.read_file(combined_path)
            build_map(
                gdf,
                title="Speed Safety Score — Maharashtra & Thailand",
                out_path=MAPS_DIR / "combined_map.html",
            )
            break
