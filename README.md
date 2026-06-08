# Speed Safety Score — AI-Powered Road Speed Limit Assessment

**ADB AI for Safer Roads Innovation Challenge 2026**  
*Team: hksamm | Pusan National University, Republic of Korea*

A five-stage multimodal AI pipeline that assesses whether posted speed limits align with WHO Safe System principles — identifying every road segment where the limit itself endangers lives, quantifying the economic cost of inaction, and producing a ranked intervention portfolio governments can act on immediately.

---

## Live Results

| Region | Segments | Grade D Unsafe | Grade C Caution | Annual Eco. Value |
|---|---|---|---|---|
| Thailand | 11,134 | **401** ($1,153.5M/yr) | 5,586 | $1,153.5M (D) |
| Maharashtra, India | 3,577 | 0 (GNN reclassified to C) | **296** ($987.3M/yr) | $987.3M (C) |

**Most dangerous segment:** Urban primary road in Thailand — 90 km/h posted limit, 115.5 km/h operating speed (85th pct), Safe System threshold 50 km/h. Nilsson Power Model: **82% fewer fatalities** if corrected to threshold.

**Maharashtra amplifier:** Only 1.2% of motorcycle passengers wear helmets. At any speed on a Grade D segment, a crash is almost certainly fatal.

---

## Interactive Dashboard

> **Live demo:** run locally in one command:
> ```bash
> pip install streamlit streamlit-folium plotly
> streamlit run app.py
> ```

The Streamlit dashboard provides:
- **Interactive map** — click any segment to see grade, CI badge, speed profile, economic value, archetype, intervention
- **Priority ranking table** — filterable by region, grade, archetype; download as CSV
- **Portfolio optimiser** — visual budget scenario analysis (10–500 segments)
- **Archetype explorer** — distribution charts and intervention guidance per cluster
- **Methodology reference** — pipeline diagram, thresholds, Nilsson formula

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: Safe System Tabular Scorer              (CPU / Mac)    │
│  → WHO/UNECE thresholds × road class × land use                  │
│  → 5 sub-scores → Speed Safety Score 0–100 → Grade A–E          │
│  → Nilsson Power Model: fatality reduction if limit corrected    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Stage 2a: Mapillary Image Fetch               (CPU, 32 threads) │
│  → Progressive radius fallback: 50m → 150m → 500m               │
│  → ~8,900 images fetched for high-risk segments                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Stage 2b: Qwen2-VL-72B VLM Inference         (6× RTX A5000)    │
│  → HuggingFace device_map=auto, tensor parallel across 6 GPUs   │
│  → 7 safety features extracted per image as structured JSON:    │
│    pedestrian_infra · cyclist_infra · roadside_activity         │
│    road_condition · signage_quality · vru_exposure · visibility  │
│  → 4,195 segments in 26 minutes                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Stage 2c: YOLOv8-L Object Detection          (1× RTX A5000)    │
│  → Counts pedestrians, cyclists, motorcycles per image           │
│  → Computes VRU density ratio per segment                        │
│  → Objective quantification replacing subjective VLM estimate    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Stage 3: Graph Attention Network             (GPU cluster)      │
│  → Road segments as nodes, junctions as edges (20m snap)        │
│  → 3-layer GAT, 4 attention heads, skip connections              │
│  → 300 epochs, CosineAnnealingLR, final RMSE 2.46               │
│  → Monte Carlo Dropout: 50 passes → 95% CI per segment          │
│  → Grade-uncertain flag when CI spans a grade boundary           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Final Score = 0.45×(Stage1) + 0.25×(GNN) + 0.30×(VLM)         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Post-Scoring Analysis                        (CPU / Mac)        │
│  → Economic impact: VOSL × Nilsson × VMT proxy → USD/year       │
│  → Segment archetype clustering (KMeans, 5 types)               │
│  → Greedy intervention portfolio optimiser                       │
│  → Ranked priority lists (global + regional)                     │
│  → Per-segment policy briefs (Claude claude-opus-4-8 API, ~250 words each) │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  Interactive Maps (Folium → GitHub Pages HTML)                   │
│  → Grade A–E choropleth with per-segment popups                  │
│  → Popup: speed profile · CI badge · economic value             │
│           VLM feature bars · YOLO counts · archetype tag        │
│           priority rank · recommended intervention               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Speed Safety Score

### Grade System

| Grade | Score | Label | Meaning |
|---|---|---|---|
| A | 0–20 | Safe | Limit aligns with Safe System principles |
| B | 20–40 | Adequate | Minor misalignment — monitor |
| C | 40–60 | Caution | Limit review recommended within 2–3 years |
| D | 60–80 | Unsafe | Limit reduction recommended urgently |
| E | 80–100 | Critical | Immediate intervention required |

### Sub-Score Components

| Component | Weight | What it measures |
|---|---|---|
| Speed deviation | **40%** | How far 85th-pct operating speed exceeds Safe System threshold |
| Posted limit excess | **20%** | How far the posted limit itself exceeds the Safe System threshold |
| Speeding prevalence | **20%** | % of vehicles exceeding the posted limit (TomTom probe data) |
| Traffic exposure | **10%** | Traffic volume — more exposure = more lives at risk |
| VRU vulnerability | **10%** | Inverse helmet-wearing rate — unprotected road users amplify severity |

### Safe System Speed Thresholds (km/h)

| Road class | Urban | Rural |
|---|---|---|
| Motorway | 80 | 110 |
| Trunk | 60 | 80 |
| Primary | 50 | 80 |
| Secondary | 40 | 60 |

*Source: WHO Safe System approach, UNECE Resolution on Road Safety Targets*

### Final Score Fusion

```
# With VLM imagery (71.5% of high-risk segments):
Final = 0.45 × Stage1 + 0.25 × GNN + 0.30 × (VLM_mean × 100)

# Without VLM imagery (fallback):
Final = 0.60 × Stage1 + 0.40 × GNN
```

### Counterfactual Impact — Nilsson's Power Model

```
Fatality_reduction = 1 − (v_safe / v_85th_pct)⁴
```

Example: 100 km/h → 50 km/h: 1 − (50/100)⁴ = **93.75% fewer fatalities**

### Uncertainty Quantification

Monte Carlo Dropout (50 stochastic forward passes) produces per-segment 95% confidence intervals. Segments where the CI crosses a grade boundary (at 20/40/60/80 points) are flagged with a warning — the true grade is ambiguous and the segment should be prioritised for on-the-ground review.

---

## Economic Impact Analysis

Annual economic value of correcting each unsafe segment:

```
Annual VMT proxy = estimated_daily_traffic(RankedPercentile) × length_km × 365
Fatal crashes averted = (VMT / 100M) × crash_rate × Nilsson_reduction
Economic benefit (USD) = crashes_averted × VOSL
```

| Country | Value of Statistical Life | Fatal crash rate |
|---|---|---|
| Thailand | $1.26M USD (World Bank 2023) | 8.4 per 100M VMT |
| Maharashtra | $0.42M USD (India MoRTH IRC:SP:88) | 11.2 per 100M VMT |

*These are relative indices for ranking interventions — not calibrated absolute predictions.*

---

## Segment Archetypes

Grade C/D/E segments are clustered by KMeans into five risk archetypes, each with tailored intervention guidance:

| Archetype | Defining signature | Primary intervention |
|---|---|---|
| 🏙️ Urban Speedway | High posted limit, urban primary road, policy misalignment | Speed limit policy reform + average speed enforcement |
| 🚗 High-Volume Corridor | Borderline excess but extreme traffic exposure | Peak-hour variable speed limits + congestion-responsive enforcement |
| 🚶 Infrastructure Void | High speeds + absent pedestrian/cyclist protection | Physical separation: bridges, barriers, protected crossings |
| ⚡ Speed Creep Zone | Acceptable limit, high non-compliance rate | Speed camera deployment + signage visibility audit |
| 🌾 Rural Risk Corridor | Rural alignment, high VRU presence, no separation | Roadside safety treatment + collision warning systems |

---

## Repository Structure

```
src/
  config.py                 # All constants: thresholds, VOSL, archetype profiles
  preprocessing/
    load_data.py            # Load & clean GeoJSON datasets
  scoring/
    safe_system.py          # Stage 1: tabular scorer + Nilsson + economic impact
  vision/
    mapillary.py            # Mapillary API client + batch fetcher
    vlm_inference.py        # Stage 2b: Qwen2-VL inference (7B or 72B)
  gnn/
    graph_builder.py        # Build road network graph + feature matrix
    model.py                # Stage 3: GAT + MC Dropout uncertainty
  analysis/
    clustering.py           # KMeans segment archetype discovery
    optimizer.py            # Greedy intervention portfolio optimiser
  visualization/
    dashboard.py            # Interactive Folium map generator

scripts/
  run_stage1.py             # Stage 1: tabular scoring
  run_stage2_fetch.py       # Fetch Mapillary imagery
  run_stage2_vlm_72b.py     # VLM inference: 72B model, 6× GPU
  run_stage2_vlm_parallel.py# VLM inference: 7B model, data parallel
  run_object_detection.py   # YOLOv8 VRU counting on street images  ← NEW
  run_stage3_gnn.py         # GNN training + MC Dropout uncertainty
  run_analysis.py           # Post-GNN: economic + archetypes + optimizer  ← NEW
  generate_priority_list.py # Export ranked intervention CSV  ← NEW
  generate_policy_briefs.py # Per-segment briefs via Claude API  ← NEW
  generate_maps.py          # Produce final interactive HTML maps

outputs/
  scores/                   # Scored GeoJSON files (stage1, final)
  maps/                     # Interactive HTML maps
  priority/                 # Ranked intervention CSV files  ← NEW
  policy_briefs/            # Per-segment engineering assessments  ← NEW
  analysis/                 # Archetype summaries, portfolio scenarios  ← NEW

report/
  findings_summary.md       # 5-page findings for submission
  submission_form_text.md   # Copy-paste text for ADB submission form
```

---

## Running the Full Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
# On GPU server also:
pip install transformers accelerate Pillow ultralytics anthropic
```

### Step 1 — Tabular Scoring (Mac / any CPU)

```bash
python scripts/run_stage1.py --region all
```

### Step 2a — Fetch Mapillary Imagery

```bash
export MAPILLARY_TOKEN="MLY|..."
# Fetch for ALL grades (most comprehensive):
python scripts/run_stage2_fetch.py --region all --threads 32 --grades A,B,C,D,E
```

### Step 2b — VLM Inference (GPU server — 72B across 6 GPUs)

```bash
# On server, 6 free GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 scripts/run_stage2_vlm_72b.py
```

### Step 2c — Object Detection (GPU server — 1 GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_object_detection.py --model l
```

### Step 3 — GNN + Uncertainty (GPU server)

```bash
python3 scripts/run_stage3_gnn.py --region all --epochs 300 --device cuda
```

### Sync to Mac

```bash
rsync -avz -e "ssh -p 8022" \
  GenAI_202455474@164.125.18.141:~/adb-safer-roads/outputs/scores/ \
  "./outputs/scores/"
```

### Post-GNN Analysis (Mac)

```bash
# Economic impact + archetypes + optimizer + priority ranks
python scripts/run_analysis.py

# Export ranked intervention CSVs
python scripts/generate_priority_list.py

# Generate policy briefs (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_policy_briefs.py --grade D,E

# Rebuild interactive maps
python scripts/generate_maps.py
```

---

## Intervention Portfolio Scenarios

The greedy optimizer answers: *"Given a budget to review N segments, which N save the most lives per dollar spent?"*

| Portfolio | Segments | Expected eco. benefit | Avg Nilsson reduction |
|---|---|---|---|
| Emergency (10) | Top 10 Grade D/E | highest per-segment impact | >60% |
| Priority (50) | Top 50 Grade C/D/E | — | — |
| Programme (100) | Top 100 | — | — |
| Full (500) | Top 500 | — | — |

*Run `python scripts/run_analysis.py` to compute exact numbers for the dataset.*

---

## Outputs for Judges

| File | Description |
|---|---|
| `outputs/maps/thailand_map.html` | Interactive map — Thailand |
| `outputs/maps/maharashtra_map.html` | Interactive map — Maharashtra |
| `outputs/maps/combined_map.html` | Both regions combined |
| `outputs/priority/priority_list_all.csv` | All risk segments, globally ranked |
| `outputs/priority/top10_emergency.csv` | Top-10 emergency interventions |
| `outputs/priority/top50_portfolio.csv` | Optimal 50-segment portfolio |
| `outputs/policy_briefs/executive_summary.md` | Top-10 segments, full briefs |
| `outputs/policy_briefs/combined_briefs_thailand.md` | All Thailand Grade D/E briefs |
| `outputs/analysis/portfolio_scenarios.json` | Budget scenario analysis |
| `report/findings_summary.md` | 5-page findings document |

---

## Scalability to Other Countries

The pipeline requires three data inputs available globally:

| Input | Global availability | Free alternative |
|---|---|---|
| Road network + classification | Overture Maps (monthly, global) | OpenStreetMap |
| Operating speeds + posted limits | TomTom Move (commercial) | HERE, Google Roads API |
| Street imagery | Mapillary (crowdsourced, global) | Google Street View Static API |

**Stage 1 alone** produces actionable Speed Safety Scores without GPU infrastructure. The VLM and GNN stages add precision and spatial context but are not prerequisites for policy use.

In data-scarce environments, 85th-percentile speeds can be estimated from road geometry and land use using regression models trained on the Thailand/Maharashtra data — enabling zero-cost transfer.

---

## Data Sources

| Source | Description |
|---|---|
| TomTom Move (via ADB) | GPS probe: 85th-pct speed, posted limit, % exceeding limit, traffic volume |
| Overture Maps | Road network: functional class, segment geometry |
| NASA GRUMP | Urban/rural land use classification |
| Mapillary | Street-level imagery (public API) |
| ADB Road Safety SPI | Helmet-wearing rates by region, land use, road user type |

## Attribution

Road network: © OpenStreetMap contributors, Overture Maps Foundation (ODbL)  
GPS probe data: TomTom Move (via ADB Challenge dataset)  
Land use: NASA GRUMP  
Street imagery: Mapillary (CC-BY-SA)  
VLM: Qwen2-VL-72B-Instruct (Alibaba Cloud / HuggingFace)  
Object detection: YOLOv8 (Ultralytics, AGPL-3.0)  
Policy briefs: Claude claude-opus-4-8 (Anthropic API)
