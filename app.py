"""
Speed Safety Score — Interactive Dashboard
ADB AI for Safer Roads Innovation Challenge 2026

Run locally:
    pip install streamlit folium streamlit-folium plotly
    streamlit run app.py

Or visit the live deployment at the Streamlit Cloud link in the README.
"""
import json
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speed Safety Score — ADB Challenge 2026",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PRIORITY_DIR = ROOT / "outputs" / "priority"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"

GRADE_COLORS = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c", "E": "#8e0000"}
GRADE_LABELS = {"A": "Safe", "B": "Adequate", "C": "Caution", "D": "Unsafe", "E": "Critical"}
ARCHETYPE_ICONS = {
    "Urban Speedway": "🏙️",
    "High-Volume Corridor": "🚗",
    "Infrastructure Void": "🚶",
    "Speed Creep Zone": "⚡",
    "Rural Risk Corridor": "🌾",
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    all_path = PRIORITY_DIR / "priority_list_all.csv"
    de_path  = PRIORITY_DIR / "priority_list_DE_only.csv"
    port_path = ANALYSIS_DIR / "portfolio_scenarios.json"

    df_all = pd.read_csv(all_path) if all_path.exists() else pd.DataFrame()
    df_de  = pd.read_csv(de_path)  if de_path.exists()  else pd.DataFrame()

    portfolios = {}
    if port_path.exists():
        with open(port_path) as f:
            portfolios = json.load(f)

    return df_all, df_de, portfolios


df_all, df_de, portfolios = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/ADB_logo_matte.svg/320px-ADB_logo_matte.svg.png", width=160)
st.sidebar.title("Speed Safety Score")
st.sidebar.caption("ADB AI for Safer Roads 2026 · hksamm · Pusan National University")
st.sidebar.markdown("---")

region_opts = ["All regions"] + sorted(df_all["region"].dropna().unique().tolist()) if not df_all.empty else ["All regions"]
region_sel  = st.sidebar.selectbox("Region", region_opts)

grade_opts = ["All grades"] + ["A", "B", "C", "D", "E"]
grade_sel  = st.sidebar.multiselect("Grade filter", ["A", "B", "C", "D", "E"], default=["C", "D", "E"])

archetype_opts = df_de["archetype"].dropna().unique().tolist() if not df_de.empty else []
arch_sel = st.sidebar.multiselect("Archetype filter", archetype_opts, default=archetype_opts)

st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline stages**")
st.sidebar.markdown(
    "1️⃣ Safe System tabular scorer  \n"
    "2️⃣ Qwen2-VL-72B vision analysis  \n"
    "2️⃣ YOLOv8-L object detection  \n"
    "3️⃣ Graph Attention Network  \n"
    "📊 MC Dropout uncertainty  \n"
    "💰 Economic impact (VOSL × Nilsson)  \n"
    "🗂️ KMeans archetype clustering  \n"
    "🎯 Portfolio optimiser"
)

# ── Filter df ─────────────────────────────────────────────────────────────────
def apply_filters(df):
    if df.empty:
        return df
    if region_sel != "All regions":
        df = df[df["region"].str.lower() == region_sel.lower()]
    if grade_sel:
        df = df[df["final_grade"].isin(grade_sel)]
    return df


filtered_all = apply_filters(df_all.copy()) if not df_all.empty else pd.DataFrame()
filtered_de  = df_de[df_de["archetype"].isin(arch_sel)].copy() if not df_de.empty and arch_sel else df_de.copy()
if region_sel != "All regions" and not filtered_de.empty:
    filtered_de = filtered_de[filtered_de["region"].str.lower() == region_sel.lower()]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.title("🛣️ Speed Safety Score — ADB Innovation Challenge 2026")
st.markdown(
    "A five-stage multimodal AI pipeline assessing whether posted speed limits align with "
    "**WHO Safe System principles** — identifying every road segment where the limit itself endangers lives."
)
st.markdown("---")

# ── Metric row ────────────────────────────────────────────────────────────────
if not df_all.empty:
    total_segs  = len(df_all)
    grade_d_e   = len(df_all[df_all["final_grade"].isin(["D", "E"])])
    eco_total_m = df_de["economic_benefit_usd_annual"].sum() / 1e6 if not df_de.empty else 0
    uncertain   = df_de["grade_uncertain"].sum() if "grade_uncertain" in df_de.columns else 0
    nilsson_max = df_de["nilsson_fatality_reduction_pct"].max() if not df_de.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total segments", f"{total_segs:,}")
    c2.metric("Grade D (Unsafe)", f"{grade_d_e:,}", delta=f"{grade_d_e/total_segs*100:.1f}% of network", delta_color="inverse")
    c3.metric("Economic value (D/E)", f"${eco_total_m:.0f}M/yr", help="Annual USD benefit if all Grade D/E limits corrected (VOSL × Nilsson)")
    c4.metric("Grade-uncertain segs", f"{int(uncertain)}", help="95% CI spans a grade boundary → prioritise for on-ground review")
    c5.metric("Max Nilsson reduction", f"{nilsson_max:.0f}%", help="Fatality reduction if highest-risk segment corrected to Safe System threshold")
    st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_rank, tab_portfolio, tab_archetype, tab_methodology = st.tabs([
    "🗺️ Interactive Map",
    "📋 Priority Ranking",
    "💼 Portfolio Optimiser",
    "🏷️ Archetypes",
    "📐 Methodology",
])

# ══════════════════ TAB 1: MAP ═════════════════════════════════════════════════
with tab_map:
    st.subheader("Segment Risk Map")
    st.caption(
        "Each marker is one road segment. Color = grade (green→red). "
        "Click any marker for full analysis details."
    )

    map_df = filtered_de if not filtered_de.empty else df_de
    if filtered_all is not None and not filtered_all.empty and "D" not in grade_sel:
        map_df = filtered_all

    # Use DE data for map (has lat/lon); fall back to all
    plot_df = filtered_de.copy() if not filtered_de.empty else pd.DataFrame()
    if region_sel != "All regions" and not df_all.empty:
        plot_df = df_all[df_all["region"].str.lower() == region_sel.lower()].copy()
        if grade_sel:
            plot_df = plot_df[plot_df["final_grade"].isin(grade_sel)]

    # Build folium map
    if not plot_df.empty and "latitude" in plot_df.columns:
        center_lat = plot_df["latitude"].median()
        center_lon = plot_df["longitude"].median()
    else:
        center_lat, center_lon = 13.7, 100.5  # Bangkok default

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")
    cluster = MarkerCluster(max_cluster_radius=40).add_to(m)

    DISPLAY_LIMIT = 2000
    plot_sample = plot_df.head(DISPLAY_LIMIT) if not plot_df.empty else pd.DataFrame()

    for _, row in plot_sample.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        grade = str(row.get("final_grade", "?"))
        color = GRADE_COLORS.get(grade, "#aaa")
        label = GRADE_LABELS.get(grade, "Unknown")
        name  = row.get("road_name") or "Unnamed segment"
        arch  = row.get("archetype", "")
        arch_icon = ARCHETYPE_ICONS.get(arch, "")
        eco   = row.get("economic_benefit_usd_annual", 0) or 0
        nilsson = row.get("nilsson_fatality_reduction_pct", 0) or 0
        ci_low  = row.get("score_ci_low", "")
        ci_high = row.get("score_ci_high", "")
        uncertain_flag = "⚠️ Grade uncertain" if row.get("grade_uncertain") else ""
        score   = row.get("final_score", 0) or 0
        prank   = row.get("priority_rank_global", "")

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:280px;max-width:340px">
          <div style="background:{color};color:white;padding:8px 12px;border-radius:4px 4px 0 0">
            <b>Grade {grade} — {label}</b>
            {"&nbsp;&nbsp;<span style='font-size:11px'>⚠️ uncertain</span>" if row.get("grade_uncertain") else ""}
          </div>
          <div style="padding:10px 12px;border:1px solid #ddd;border-top:none;border-radius:0 0 4px 4px">
            <b>{name[:50]}</b><br>
            <span style="color:#666;font-size:12px">{row.get('road_class','?').title()} · {row.get('land_use','?').title()} · {row.get('region','?').title()}</span>
            <hr style="margin:6px 0">
            <table style="width:100%;font-size:12px">
              <tr><td>Final score</td><td><b>{score:.1f}</b> [{ci_low} – {ci_high}] / 100</td></tr>
              <tr><td>Posted limit</td><td>{row.get('posted_limit_kmh','?')} km/h</td></tr>
              <tr><td>85th pct speed</td><td>{row.get('speed_85th_pct_kmh','?')} km/h</td></tr>
              <tr><td>Safe System threshold</td><td>{row.get('safe_system_threshold_kmh','?')} km/h</td></tr>
              <tr><td>Nilsson reduction</td><td><b>{nilsson:.1f}%</b> fewer fatalities if corrected</td></tr>
              <tr><td>Economic value</td><td><b>${eco:,.0f}/yr</b></td></tr>
              <tr><td>Global priority rank</td><td>#{prank}</td></tr>
            </table>
            <hr style="margin:6px 0">
            <span style="font-size:12px">{arch_icon} <b>{arch}</b></span><br>
            <span style="font-size:11px;color:#444">Primary: {row.get('primary_intervention','')[:60]}</span>
          </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"Grade {grade} | {name[:40]} | ${eco:,.0f}/yr",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(cluster)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:9999;background:white;
                padding:12px 16px;border-radius:8px;border:1px solid #ddd;font-family:sans-serif;font-size:13px">
      <b>Speed Safety Grade</b><br>
      <span style="color:#2ecc71">●</span> A — Safe<br>
      <span style="color:#f1c40f">●</span> B — Adequate<br>
      <span style="color:#e67e22">●</span> C — Caution<br>
      <span style="color:#e74c3c">●</span> D — Unsafe<br>
      <span style="color:#8e0000">●</span> E — Critical
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if len(plot_sample) < len(plot_df):
        st.info(f"Showing top {DISPLAY_LIMIT:,} segments of {len(plot_df):,} matching filters.")

    st_folium(m, width=None, height=600, returned_objects=[])


# ══════════════════ TAB 2: PRIORITY RANKING ═══════════════════════════════════
with tab_rank:
    st.subheader("Priority Ranking — Highest-Impact Interventions")

    show_df = filtered_de.copy() if not filtered_de.empty else df_de.copy()

    col_display = [
        "priority_rank_global", "region", "road_name", "road_class", "land_use",
        "archetype", "final_grade", "final_score", "score_ci_low", "score_ci_high",
        "grade_uncertain", "posted_limit_kmh", "speed_85th_pct_kmh",
        "safe_system_threshold_kmh", "nilsson_fatality_reduction_pct",
        "economic_benefit_usd_annual", "primary_intervention",
    ]
    col_display = [c for c in col_display if c in show_df.columns]

    if not show_df.empty:
        show_df_display = show_df[col_display].sort_values("priority_rank_global")
        show_df_display["economic_benefit_usd_annual"] = show_df_display["economic_benefit_usd_annual"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else ""
        )
        show_df_display["grade_uncertain"] = show_df_display["grade_uncertain"].apply(
            lambda x: "⚠️" if x else ""
        )
        st.dataframe(
            show_df_display.rename(columns={
                "priority_rank_global": "Global Rank",
                "region": "Region",
                "road_name": "Road Name",
                "road_class": "Class",
                "land_use": "Land Use",
                "archetype": "Archetype",
                "final_grade": "Grade",
                "final_score": "Score",
                "score_ci_low": "CI Low",
                "score_ci_high": "CI High",
                "grade_uncertain": "Uncertain",
                "posted_limit_kmh": "Posted (km/h)",
                "speed_85th_pct_kmh": "85th pct (km/h)",
                "safe_system_threshold_kmh": "Threshold (km/h)",
                "nilsson_fatality_reduction_pct": "Nilsson %",
                "economic_benefit_usd_annual": "Eco. Value/yr",
                "primary_intervention": "Primary Intervention",
            }),
            use_container_width=True,
            height=500,
        )

        st.download_button(
            "⬇️ Download filtered priority list (CSV)",
            data=show_df.to_csv(index=False),
            file_name="priority_list_filtered.csv",
            mime="text/csv",
        )

    # Economic value chart
    if not show_df.empty and "economic_benefit_usd_annual" in show_df.columns:
        top20 = show_df.nsmallest(20, "priority_rank_global")[
            ["road_name", "region", "economic_benefit_usd_annual", "final_grade", "nilsson_fatality_reduction_pct"]
        ].copy()
        top20["economic_benefit_m"] = top20["economic_benefit_usd_annual"] / 1e6
        top20["label"] = top20["road_name"].str[:30] + " (" + top20["region"].str.title() + ")"

        fig = px.bar(
            top20.sort_values("economic_benefit_m"),
            x="economic_benefit_m",
            y="label",
            color="final_grade",
            color_discrete_map=GRADE_COLORS,
            orientation="h",
            labels={"economic_benefit_m": "Annual economic value (USD million)", "label": ""},
            title="Top 20 segments by annual economic impact of intervention",
        )
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════ TAB 3: PORTFOLIO OPTIMISER ════════════════════════════════
with tab_portfolio:
    st.subheader("Intervention Portfolio Optimiser")
    st.markdown(
        "Given a finite budget to review **N segments**, the greedy optimiser selects the set that "
        "maximises **economic benefit per unit of review cost** — answering the policy question: "
        "*where do we act first?*"
    )

    if portfolios:
        port_keys = sorted(portfolios.keys(), key=lambda x: int(x))
        port_data = []
        for k in port_keys:
            p = portfolios[k]
            port_data.append({
                "Budget (segments)": int(k),
                "Segments selected": p.get("n_segments", int(k)),
                "Total eco. value (M USD/yr)": round(p.get("total_economic_benefit_usd", 0) / 1e6, 1),
                "Avg Nilsson reduction (%)": round(p.get("avg_nilsson_reduction", 0), 1),
                "Avg score": round(p.get("avg_final_score", 0), 1),
            })
        port_df = pd.DataFrame(port_data)
        st.dataframe(port_df, use_container_width=True, hide_index=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=port_df["Budget (segments)"],
            y=port_df["Total eco. value (M USD/yr)"],
            name="Economic value (M USD/yr)",
            marker_color="#e74c3c",
        ))
        fig2.add_trace(go.Scatter(
            x=port_df["Budget (segments)"],
            y=port_df["Avg Nilsson reduction (%)"],
            name="Avg Nilsson % (right axis)",
            yaxis="y2",
            mode="lines+markers",
            marker_color="#3498db",
        ))
        fig2.update_layout(
            title="Portfolio value vs. number of segments reviewed",
            xaxis_title="Number of segments in portfolio",
            yaxis_title="Total annual economic value (M USD)",
            yaxis2=dict(title="Avg Nilsson fatality reduction (%)", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "**Interpretation:** Reviewing just the top **10 segments** captures a "
            "disproportionate share of total economic value — consistent with the "
            "highly concentrated nature of speed-related risk on road networks."
        )
    else:
        st.info("Portfolio data not found. Run `python scripts/run_analysis.py` to generate.")

    # Top 10 emergency list
    top10_path = PRIORITY_DIR / "top10_emergency.csv"
    if top10_path.exists():
        st.subheader("Top 10 Emergency Interventions")
        t10 = pd.read_csv(top10_path)
        display_cols = [c for c in ["priority_rank_global", "region", "road_name", "road_class", "land_use",
                                     "final_grade", "final_score", "nilsson_fatality_reduction_pct",
                                     "economic_benefit_usd_annual", "archetype", "primary_intervention"] if c in t10.columns]
        st.dataframe(t10[display_cols], use_container_width=True, hide_index=True)


# ══════════════════ TAB 4: ARCHETYPES ═════════════════════════════════════════
with tab_archetype:
    st.subheader("Segment Risk Archetypes")
    st.markdown(
        "KMeans clustering (k=5) groups Grade C/D/E segments into five distinct risk patterns "
        "with tailored intervention strategies — matching the right tool to each road type."
    )

    arch_path = ANALYSIS_DIR / "combined_archetype_summary.csv"
    if arch_path.exists():
        arch_df = pd.read_csv(arch_path)

        for _, row in arch_df.iterrows():
            arch_name = row.get("archetype", "")
            icon = row.get("icon", ARCHETYPE_ICONS.get(arch_name, ""))
            with st.expander(f"{icon} **{arch_name}** — {row.get('count', 0)} segments", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Segments", int(row.get("count", 0)))
                c2.metric("Avg score", f"{row.get('avg_final_score', 0):.1f}")
                c3.metric("Total eco. value", f"${row.get('total_economic_benefit_m', 0):.1f}M/yr")
                c4.metric("Avg Nilsson", f"{row.get('avg_nilsson_pct', 0):.1f}%")
                st.markdown(f"**Description:** {row.get('description', '')}")
                st.markdown(f"**Primary intervention:** {row.get('intervention', '')}")
    else:
        st.info("Run `python scripts/run_analysis.py` to generate archetype summaries.")

    # Archetype distribution chart
    if not df_de.empty and "archetype" in df_de.columns:
        arch_counts = df_de.groupby(["region", "archetype"]).size().reset_index(name="count")
        fig3 = px.bar(
            arch_counts,
            x="archetype",
            y="count",
            color="region",
            barmode="group",
            title="Grade D segment archetype distribution by region",
            labels={"count": "Number of segments", "archetype": ""},
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Grade distribution by archetype
    if not df_de.empty:
        eco_by_arch = df_de.groupby("archetype")["economic_benefit_usd_annual"].sum().reset_index()
        eco_by_arch["economic_benefit_m"] = eco_by_arch["economic_benefit_usd_annual"] / 1e6
        fig4 = px.pie(
            eco_by_arch,
            values="economic_benefit_m",
            names="archetype",
            title="Economic value distribution by archetype (M USD/yr)",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════ TAB 5: METHODOLOGY ═══════════════════════════════════════
with tab_methodology:
    st.subheader("Pipeline Methodology")

    st.markdown("""
### Five-Stage Multimodal AI Pipeline

| Stage | Description | Compute |
|---|---|---|
| **1 — Tabular Scorer** | WHO Safe System thresholds × road class × land use → Speed Safety Score 0–100 | CPU |
| **2a — Mapillary Fetch** | Street-level imagery for all 14,711 segments via Mapillary Graph API | CPU, 32 threads |
| **2b — Qwen2-VL-72B** | 7 safety features extracted per image as structured JSON | 6× RTX A5000 GPU |
| **2c — YOLOv8-L** | Objective VRU counting (pedestrians, cyclists, motorcycles) per image | 1× RTX A5000 GPU |
| **3 — GAT + MC Dropout** | 3-layer Graph Attention Network, 300 epochs; 50 stochastic forward passes → 95% CI | GPU cluster |

### Safe System Speed Thresholds (km/h)

| Road class | Urban | Rural |
|---|---|---|
| Motorway | 80 | 110 |
| Trunk | 60 | 80 |
| Primary | 50 | 80 |
| Secondary | 40 | 60 |

### Final Score Fusion

```
With VLM imagery:   Final = 0.45 × Stage1 + 0.25 × GNN + 0.30 × (VLM_mean × 100)
Without imagery:    Final = 0.60 × Stage1 + 0.40 × GNN
```

### Nilsson Power Model

```
Fatality_reduction = 1 − (v_safe / v_85th_pct)⁴

Example: 100 km/h → 50 km/h = 1 − (50/100)⁴ = 93.75% fewer fatalities
```

### Economic Impact

```
Annual VMT proxy  = estimated_daily_traffic(RankedPercentile) × length_km × 365
Crashes averted   = (VMT / 100M) × crash_rate_per_100M_VMT × Nilsson_reduction
Economic benefit  = crashes_averted × VOSL (Value of Statistical Life)
```

| Country | VOSL | Crash rate |
|---|---|---|
| Thailand | $1.26M USD (World Bank 2023) | 8.4 per 100M VMT |
| Maharashtra | $0.42M USD (India MoRTH IRC:SP:88) | 11.2 per 100M VMT |

### Uncertainty Quantification

Monte Carlo Dropout runs 50 stochastic forward passes through the trained GNN at inference time,
producing per-segment 95% confidence intervals. Segments where the CI crosses a grade boundary
(at 20/40/60/80 score points) are flagged `grade_uncertain = True` — the true grade is ambiguous
and the segment should be prioritised for on-the-ground verification.

---

*Compute: 8× NVIDIA RTX A5000 (192 GB VRAM), Pusan National University GenAI Lab*
*Full source code: [github.com/Htet-Kaung-San/adb-safer-roads](https://github.com/Htet-Kaung-San/adb-safer-roads)*
    """)
