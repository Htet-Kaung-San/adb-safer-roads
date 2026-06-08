"""
Generate a lightweight interactive map showing only Grade D (Unsafe) segments.
Designed to be small enough to commit to GitHub (<5MB).

Output: outputs/maps/priority_grade_D_map.html
"""
import sys
from pathlib import Path
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

ROOT = Path(__file__).parent.parent

def grade_color(grade):
    return {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c", "E": "#8e0000"}.get(grade, "#999")

def load_region(path, region_name):
    gdf = gpd.read_file(path)
    gdf["region"] = region_name
    return gdf

def make_priority_map():
    print("Loading data...")
    th = load_region(ROOT / "outputs/scores/thailand_final.geojson", "thailand")
    mh = load_region(ROOT / "outputs/scores/maharashtra_final.geojson", "maharashtra")

    # Thailand Grade D + Maharashtra Grade C (the most critical segments)
    th_d = th[th["final_grade"] == "D"].copy()
    mh_c = mh[mh["final_grade"] == "C"].copy()
    print(f"Thailand Grade D: {len(th_d)}, Maharashtra Grade C: {len(mh_c)}")

    # Center on Southeast/South Asia
    m = folium.Map(
        location=[18.0, 95.0],
        zoom_start=5,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # -- Thailand Grade D cluster --
    th_cluster = MarkerCluster(name="Thailand Grade D (Unsafe)", show=True).add_to(m)

    for _, row in th_d.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lat = geom.centroid.y
            lon = geom.centroid.x
        except Exception:
            continue

        eco = row.get("economic_benefit_usd", 0) or 0
        nilsson = row.get("nilsson_fatality_reduction_pct", 0) or 0
        score = row.get("final_score", 0) or 0
        name = str(row.get("road_name", "Unnamed"))
        prank = row.get("priority_rank_global", "")
        ci_low  = row.get("score_ci_low", "")
        ci_high = row.get("score_ci_high", "")
        arch = str(row.get("archetype", ""))
        pop  = row.get("population_per_km2", None)
        crashes = row.get("crash_count", None)
        osm_infra = row.get("osm_infrastructure_score", None)
        osm_sw = row.get("osm_has_sidewalk", None)

        def osm_bool(v):
            if v is None or (isinstance(v, float) and v != v): return "—"
            return "Yes" if str(v).lower() in ("true", "1") else "No"

        osm_section = ""
        if osm_infra is not None and not (isinstance(osm_infra, float) and osm_infra != osm_infra):
            osm_maxspeed = row.get("osm_maxspeed_kmh", None)
            osm_lanes = row.get("osm_lanes", None)
            osm_lit = row.get("osm_lit", None)
            osm_section = f"""
            <tr style="background:#fef2f2"><td colspan="2"><b>OSM Ground Truth</b></td></tr>
            <tr><td>Sidewalk</td><td>{osm_bool(osm_sw)}</td></tr>
            <tr><td>Lit</td><td>{osm_bool(osm_lit)}</td></tr>
            <tr><td>OSM maxspeed</td><td>{f'{osm_maxspeed:.0f} km/h' if osm_maxspeed else '—'}</td></tr>
            <tr><td>Lanes</td><td>{f'{int(osm_lanes)}' if osm_lanes and not (isinstance(osm_lanes, float) and osm_lanes != osm_lanes) else '—'}</td></tr>
            <tr><td>Infra score</td><td>{osm_infra:.2f}/1.0</td></tr>"""

        popup_html = f"""
<div style="font-family:sans-serif;min-width:280px;max-width:340px">
  <div style="background:#e74c3c;color:white;padding:8px 12px;border-radius:4px 4px 0 0">
    <b>Grade D — UNSAFE</b>
    {'&nbsp;<span style="font-size:10px">⚠️ grade uncertain</span>' if row.get("grade_uncertain") else ""}
  </div>
  <div style="padding:10px 12px;border:1px solid #ddd;border-top:none">
    <b>{name[:50]}</b><br>
    <span style="font-size:11px;color:#666">{str(row.get("road_class","?")).title()} · URBAN · Thailand</span>
    <hr style="margin:6px 0">
    <table style="width:100%;font-size:12px">
      <tr><td>Score</td><td><b>{score:.1f}</b> [{ci_low}–{ci_high}] / 100</td></tr>
      <tr><td>Posted limit</td><td>{row.get("posted_limit_kmh","?")} km/h</td></tr>
      <tr><td>85th pct speed</td><td>{row.get("speed_85th_pct_kmh","?")} km/h</td></tr>
      <tr><td>Safe System threshold</td><td>{row.get("safe_system_threshold_kmh","?")} km/h</td></tr>
      <tr><td>Nilsson reduction</td><td><b>{nilsson:.1f}%</b> fewer fatalities</td></tr>
      <tr><td>Economic value</td><td><b>${eco:,.0f}/yr</b></td></tr>
      <tr><td>Population density</td><td>{f"{pop:,.0f} p/km²" if pop else "—"}</td></tr>
      <tr><td>Crash records (2019–22)</td><td>{f"{int(crashes)}" if crashes else "—"}</td></tr>
      <tr><td>Global priority rank</td><td>#{prank}</td></tr>
      {osm_section}
    </table>
    <hr style="margin:6px 0">
    <span style="font-size:11px;color:#333">🏙️ {arch}</span><br>
    <span style="font-size:10px;color:#666">{str(row.get("primary_intervention",""))[:60]}</span>
  </div>
</div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#b91c1c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            weight=2,
            tooltip=f"#{prank} | {name[:35]} | ${eco:,.0f}/yr | Nilsson {nilsson:.0f}%",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(th_cluster)

    # -- Maharashtra Grade C cluster --
    mh_cluster = MarkerCluster(name="Maharashtra Grade C (Caution)", show=True).add_to(m)

    for _, row in mh_c.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lat = geom.centroid.y
            lon = geom.centroid.x
        except Exception:
            continue

        eco = row.get("economic_benefit_usd", 0) or 0
        nilsson = row.get("nilsson_fatality_reduction_pct", 0) or 0
        score = row.get("final_score", 0) or 0
        name = str(row.get("road_name", "Unnamed"))
        prank = row.get("priority_rank_global", "")
        ci_low  = row.get("score_ci_low", "")
        ci_high = row.get("score_ci_high", "")
        arch = str(row.get("archetype", ""))
        pop  = row.get("population_per_km2", None)
        osm_infra = row.get("osm_infrastructure_score", None)
        osm_sw = row.get("osm_has_sidewalk", None)

        osm_section = ""
        if osm_infra is not None and not (isinstance(osm_infra, float) and osm_infra != osm_infra):
            osm_maxspeed = row.get("osm_maxspeed_kmh", None)
            osm_lanes = row.get("osm_lanes", None)
            osm_lit = row.get("osm_lit", None)
            osm_section = f"""
            <tr style="background:#fef3c7"><td colspan="2"><b>OSM Ground Truth</b></td></tr>
            <tr><td>Sidewalk</td><td>{osm_bool(osm_sw)}</td></tr>
            <tr><td>Lit</td><td>{osm_bool(osm_lit)}</td></tr>
            <tr><td>OSM maxspeed</td><td>{f'{osm_maxspeed:.0f} km/h' if osm_maxspeed else '—'}</td></tr>
            <tr><td>Infra score</td><td>{osm_infra:.2f}/1.0</td></tr>"""

        popup_html = f"""
<div style="font-family:sans-serif;min-width:280px;max-width:340px">
  <div style="background:#e67e22;color:white;padding:8px 12px;border-radius:4px 4px 0 0">
    <b>Grade C — CAUTION</b>
    {'&nbsp;<span style="font-size:10px">⚠️ grade uncertain</span>' if row.get("grade_uncertain") else ""}
  </div>
  <div style="padding:10px 12px;border:1px solid #ddd;border-top:none">
    <b>{name[:50]}</b><br>
    <span style="font-size:11px;color:#666">{str(row.get("road_class","?")).title()} · Maharashtra</span>
    <hr style="margin:6px 0">
    <table style="width:100%;font-size:12px">
      <tr><td>Score</td><td><b>{score:.1f}</b> [{ci_low}–{ci_high}] / 100</td></tr>
      <tr><td>Posted limit</td><td>{row.get("posted_limit_kmh","?")} km/h</td></tr>
      <tr><td>85th pct speed</td><td>{row.get("speed_85th_pct_kmh","?")} km/h</td></tr>
      <tr><td>Safe System threshold</td><td>{row.get("safe_system_threshold_kmh","?")} km/h</td></tr>
      <tr><td>Nilsson reduction</td><td><b>{nilsson:.1f}%</b> fewer fatalities</td></tr>
      <tr><td>Economic value</td><td><b>${eco:,.0f}/yr</b></td></tr>
      <tr><td>Population density</td><td>{f"{pop:,.0f} p/km²" if pop else "—"}</td></tr>
      <tr><td>Global priority rank</td><td>#{prank}</td></tr>
      {osm_section}
    </table>
    <hr style="margin:6px 0">
    <span style="font-size:11px;color:#333">🏙️ {arch}</span>
  </div>
</div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#c2410c",
            fill=True,
            fill_color="#e67e22",
            fill_opacity=0.85,
            weight=2,
            tooltip=f"#{prank} | {name[:35]} | ${eco:,.0f}/yr",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(mh_cluster)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;background:white;
                padding:14px 18px;border-radius:8px;border:1px solid #ddd;
                font-family:sans-serif;font-size:13px;max-width:260px">
      <b>Speed Safety Score — Priority Segments</b><br>
      <span style="color:#e74c3c">●</span> Grade D — Unsafe (Thailand, n=401)<br>
      <span style="color:#e67e22">●</span> Grade C — Caution (Maharashtra, n=296)<br>
      <hr style="margin:8px 0">
      <span style="font-size:11px">Click any segment for speed profile,<br>
      economic value, crash data, and OSM infrastructure cross-validation.</span><br><br>
      <span style="font-size:10px;color:#666">
        Ground-truth validated: 80,849 MOT/TRAMS crash records<br>
        Grade D fatality rate: 1.7× higher than Grade B (p<0.0001)
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title banner
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
                background:white;padding:10px 20px;border-radius:8px;border:1px solid #ddd;
                font-family:sans-serif;text-align:center">
      <b style="font-size:15px">Speed Safety Score — Priority Intervention Map</b><br>
      <span style="font-size:11px;color:#666">
        ADB AI for Safer Roads Innovation Challenge 2026 · hksamm / Pusan National University
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    out = ROOT / "outputs" / "maps" / "priority_grade_D_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out}")
    print(f"File size: {size_mb:.2f} MB")
    if size_mb < 5:
        print("✓ Small enough to commit to GitHub")
    else:
        print(f"⚠ Too large for GitHub ({size_mb:.1f} MB > 5MB limit)")


if __name__ == "__main__":
    make_priority_map()
