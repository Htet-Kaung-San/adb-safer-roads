# AI for Safer Roads — Speed Safety Score

**ADB AI for Safer Roads Innovation Challenge 2026**

A three-stage multimodal pipeline to assess whether posted speed limits align with WHO Safe System principles — identifying road segments where limits expose vulnerable road users to unacceptable risk.

## Live Maps
- [Maharashtra Speed Safety Map](outputs/maps/maharashtra_map.html)
- [Thailand Speed Safety Map](outputs/maps/thailand_map.html)
- [Combined Map](outputs/maps/combined_map.html)

## Key Results

| Region | Segments analysed | Grade D (Unsafe) | Grade E (Critical) |
|---|---|---|---|
| Maharashtra, India | 3,577 | 2 | 0 |
| Thailand | 11,134 | 829 | 1 |

The single Critical (E) segment: an urban primary road in Thailand with a **90 km/h posted limit**, operating at **115.5 km/h** (85th percentile), against a Safe System threshold of **50 km/h** — a 65.5 km/h excess. Estimated fatality reduction if reduced to threshold: **~82%**.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Safe System Tabular Scorer           (CPU / Mac)  │
│  → WHO/UNECE thresholds × road class × land use             │
│  → Speed deviation, posted limit excess, speeding rate      │
│  → Traffic exposure, VRU vulnerability (helmet SPI)         │
│  → Speed Safety Score 0–100 per segment                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  Stage 2: VLM Road Vision Analysis        (8× RTX A5000)    │
│  → Mapillary API: fetch street-level imagery per segment    │
│  → Qwen2-VL-72B (tensor parallel): extract visual features │
│    pedestrian_infra, cyclist_infra, roadside_activity,      │
│    road_condition, signage_quality, vru_exposure            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  Stage 3: Graph Attention Network         (GPU cluster)     │
│  → Road network as graph (segments=nodes, junctions=edges)  │
│  → GAT propagates risk context across connected segments    │
│  → Spatially-aware refined score                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  Final Score = 0.45×(Stage1) + 0.25×(GNN) + 0.30×(VLM)    │
│  + Nilsson Power Model: fatality reduction if limit fixed   │
│  + Interactive map: grade A–E choropleth + click popups     │
└─────────────────────────────────────────────────────────────┘
```

## Speed Safety Score Definition

**Grade A (0–20): Safe** — speed limit aligns with Safe System principles  
**Grade B (20–40): Adequate** — minor misalignment, monitoring recommended  
**Grade C (40–60): Caution** — limit review recommended  
**Grade D (60–80): Unsafe** — limit reduction recommended with urgency  
**Grade E (80–100): Critical** — immediate intervention required

### Sub-score Components

| Component | Weight | What it measures |
|---|---|---|
| Speed deviation | 40% | How far 85th-pct speed exceeds Safe System threshold for road type + land use |
| Posted limit excess | 20% | How far the *posted limit* itself exceeds the Safe System threshold |
| Speeding prevalence | 20% | % of vehicles exceeding posted limit (from TomTom probe data) |
| Traffic exposure | 10% | Traffic volume — more exposure = more lives at risk |
| VRU vulnerability | 10% | Inverse helmet-wearing rate — unprotected road users amplify severity |

### Safe System Speed Thresholds (km/h)

| Road class | Urban | Rural |
|---|---|---|
| Motorway | 80 | 110 |
| Trunk | 60 | 80 |
| Primary | 50 | 80 |
| Secondary | 40 | 60 |

*Based on WHO Safe System and UNECE Road Safety targets*

### Counterfactual Impact: Nilsson's Power Model

For each unsafe segment, we estimate the fatality reduction if the limit were reduced to the Safe System threshold:

```
Fatality_reduction = 1 − (v_new / v_old)⁴
```

Example: a road operating at 100 km/h reduced to 50 km/h → 1 − (50/100)⁴ = **93.75% fewer fatalities**.

## Data Sources

| Source | Description |
|---|---|
| TomTom Move (via ADB) | GPS probe data: operating speeds, 85th-pct speed, % exceeding limit |
| Overture Maps | Road network: functional class, geometry |
| NASA GRUMP | Urban/rural classification |
| Mapillary | Street-level imagery (via public API) |
| ADB Road Safety SPI | Helmet-wearing rates by region, land use, user type |

## Running the Pipeline

### Requirements
```bash
pip install -r requirements.txt
```

### Stage 1: Tabular scoring (runs on any machine)
```bash
python scripts/run_stage1.py --region all
```

### Stage 2a: Fetch Mapillary imagery
```bash
export MAPILLARY_TOKEN="MLY|..."
python scripts/run_stage2_fetch.py --region all --threads 32
```

### Stage 2b: VLM inference (GPU server — 7B single GPU)
```bash
python scripts/run_stage2_vlm.py --region all --model Qwen/Qwen2-VL-7B-Instruct
```

### Stage 2b: VLM inference (GPU server — 72B, 8× A5000)
```bash
python scripts/run_stage2_vlm.py --region all \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --tp 8 --vllm
```

### Stage 3: GNN spatial refinement
```bash
python scripts/run_stage3_gnn.py --region all --epochs 150 --device cuda
```

### Generate maps
```bash
python scripts/generate_maps.py
```

## Scalability to Other Countries

The methodology requires only:
1. Road network with functional class and urban/rural classification (Overture Maps: globally available)
2. GPS probe speed data (TomTom Move, HERE, or Google Roads API)
3. Mapillary street imagery (globally crowdsourced)

No ground-truth crash data is required. The Safe System thresholds are internationally established and can be adjusted per-country. Stage 1 alone produces actionable results with minimal data.

## Repository Structure

```
src/
  config.py              # Safe System thresholds, scoring weights, all constants
  preprocessing/
    load_data.py         # Load & clean GeoJSON datasets
  scoring/
    safe_system.py       # Stage 1: tabular scorer + Nilsson counterfactual model
  vision/
    mapillary.py         # Mapillary API client + batch image fetcher
    vlm_inference.py     # Stage 2: Qwen2-VL inference (7B or 72B)
  gnn/
    graph_builder.py     # Build road network graph + feature matrix
    model.py             # Stage 3: Graph Attention Network
  visualization/
    dashboard.py         # Interactive Folium map generator
scripts/
  run_stage1.py
  run_stage2_fetch.py
  run_stage2_vlm.py
  run_stage3_gnn.py
  generate_maps.py
outputs/
  scores/                # Scored GeoJSON files
  maps/                  # Interactive HTML maps
```

## Attribution

Road network: © OpenStreetMap contributors, Overture Maps Foundation (ODbL)  
GPS probe data: TomTom Move (via ADB Challenge dataset)  
Land use: NASA GRUMP  
Street imagery: Mapillary (CC-BY-SA)
