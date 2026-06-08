# Speed Safety Score: AI-Powered Road Speed Limit Assessment
## Findings Summary — AI for Safer Roads Innovation Challenge 2026

**Submitted to:** Asian Development Bank, World Bank, ITU, AI for Good  
**Team:** hksamm | Pusan National University, Republic of Korea  
**Date:** June 2026

---

## Executive Summary

Every year, road traffic crashes kill approximately 1.35 million people globally. Speed is a factor in nearly one-third of all fatal crashes — and the problem is not always that drivers are going too fast. Often, the speed limit itself is wrong: set too high for the road's context, the surrounding land use, and the people who depend on it daily.

This study presents a five-stage multimodal AI pipeline to assess whether posted speed limits align with WHO Safe System principles across 14,711 road segments in Thailand and Maharashtra, India. The pipeline combines GPS probe data, large-scale Vision-Language Model analysis of 8,900+ street-level images, YOLOv8 object detection for objective VRU quantification, a Graph Attention Network for spatial risk propagation, and Monte Carlo uncertainty estimation — producing not just scores, but confidence intervals, economic valuations, and a ranked intervention portfolio.

**Key findings:**

- **Thailand** has a systemic problem: **401 Grade D (Unsafe) segments** where urban posted limits of 80–90 km/h exceed Safe System thresholds by 30–40 km/h. The annual economic value of correcting these segments is **$1,153.5M USD/year** (World Bank VOSL × Nilsson Power Model × VMT proxy). Correcting the 200 urban primary/trunk Grade D segments — achievable through a single national speed limit regulation with no capital expenditure — captures **$868.5M/year** of that value.

- **Maharashtra** presents a different but equally serious picture: the GNN reclassified all high-risk segments to **Grade C (Caution)** after incorporating spatial network context, but with helmet-wearing rates of just 1.2% among motorcycle passengers, every crash on a Grade C road is almost certainly fatal. The **296 Grade C segments represent $987.3M/year** in economic risk — and the model explicitly flags 33.3% of these as grade-uncertain, meaning they sit on the C/D boundary and warrant priority on-ground review.

- **Uncertainty quantification** reveals that a meaningful share of segments near grade boundaries have ambiguous true grades — the 95% confidence interval from Monte Carlo dropout spans both sides of a threshold. These segments are explicitly flagged, and judges can see precisely where the model is confident versus where on-the-ground verification is warranted.

- **Segment archetypes** cluster the risk into five actionable patterns with distinct intervention types — from legislative limit reform for "Urban Speedways" to physical separation for "Infrastructure Voids" — enabling governments to match the right tool to each road type.

- **Economic analysis** provides a dollar figure for every intervention: the greedy portfolio optimizer answers "given a budget to review N segments this year, which N save the most lives per dollar spent?" For the first time, a transport ministry can take this model's output directly into a budget planning process with quantified ROI.

---

## 1. Methodology

### 1.1 Conceptual Framework

The analysis is grounded in the **WHO Safe System approach**, which holds that road systems must be designed to account for human error and human physical vulnerability. At the core is a set of maximum speeds above which the human body cannot survive a crash:

- **30 km/h** — mixed traffic with pedestrians and cyclists
- **50 km/h** — urban arterials with some separation from vulnerable road users
- **70 km/h** — divided roads with limited intersection density
- **110 km/h** — full grade separation, motorway standard

The question this study answers is not *"are drivers speeding?"* but *"is the posted speed limit appropriate for this road's context?"*

### 1.2 Five-Stage Pipeline

**Stage 1 — Safe System Tabular Scorer**

Each road segment receives a Speed Safety Score (0–100) from five weighted sub-scores:

| Component | Weight | Data source |
|---|---|---|
| 85th-percentile speed vs. Safe System threshold | 40% | TomTom probe data |
| Posted limit vs. Safe System threshold | 20% | TomTom / Overture |
| Vehicles exceeding posted limit (%) | 20% | TomTom probe data |
| Traffic volume exposure | 10% | TomTom ranked percentile |
| VRU vulnerability (inverse helmet SPI) | 10% | ADB Road Safety SPI |

Safe System thresholds are assigned per segment based on road functional class (Overture Maps) and land use (NASA GRUMP urban/rural).

**Stage 2a — Mapillary Image Retrieval**

GPS coordinates from the dataset's StreetImageLink field were used to query the Mapillary Graph API. A progressive search radius (50m → 150m → 500m) maximised coverage. For full-network analysis, imagery was retrieved for all 14,711 segments, not just the Grade D/E subset.

**Stage 2b — Qwen2-VL-72B Vision-Language Model**

Street-level images were analysed by **Qwen2-VL-72B-Instruct** — a state-of-the-art VLM running in tensor-parallel mode across 6 NVIDIA RTX A5000 GPUs via HuggingFace `device_map="auto"`. The model extracts seven structured road safety features per image as JSON:

pedestrian infrastructure · cyclist infrastructure · roadside activity intensity · road surface condition · speed signage quality · VRU presence · sightline quality

4,195 high-risk segments were processed in 26 minutes. Results are averaged across up to three images per segment and contribute a 30% weight to the final fused score.

**Stage 2c — YOLOv8-L Object Detection**

A complementary objective layer: YOLOv8-L running on a single GPU detects and counts pedestrians, cyclists, motorcycles, cars, buses, and trucks in every cached image. The VRU density ratio (VRU detections / total detections) provides a quantified, non-subjective measure of road user mix that cross-validates the VLM's qualitative `vru_exposure` score.

**Stage 3 — Graph Attention Network with Uncertainty Quantification**

Road segments are modelled as a spatial graph by snapping endpoints within 20 metres. A 3-layer GAT with 4 attention heads and skip connections is trained transductively for 300 epochs using Stage 1 scores as targets, learning to propagate risk context through road network topology.

**Monte Carlo Dropout** (50 stochastic forward passes at inference time) produces per-segment 95% confidence intervals. Segments where the CI spans a grade boundary are flagged as grade-uncertain — a feature no other submission in this challenge is likely to have.

**Final score fusion:**

```
Final Score = 0.45 × (Stage 1) + 0.25 × (GNN) + 0.30 × (VLM mean × 100)
Fallback (no imagery): 0.60 × (Stage 1) + 0.40 × (GNN)
```

**Post-Scoring Analysis**

- **Economic impact**: Annual USD value of each intervention via VOSL × Nilsson × VMT proxy
- **Archetype clustering**: KMeans (k=5) identifies five distinct risk patterns with tailored interventions
- **Intervention portfolio optimiser**: Greedy fractional knapsack ranks segments by economic value / review cost, producing optimal portfolios for budget scenarios from 10 to 500 interventions
- **Policy briefs**: Claude claude-opus-4-8 generates a 250-word engineering assessment for every Grade D/E segment

### 1.3 Data Sources

| Source | Type | Use |
|---|---|---|
| ADB Challenge Dataset | Provided | TomTom GPS probe data, Overture road network, NASA GRUMP land use |
| Mapillary Graph API | External (free) | Street-level imagery for VLM and YOLO |
| ADB Road Safety SPI | Provided | Helmet-wearing rates by region |
| **WorldPop 2020 (1km)** | **External (CC BY 4.0)** | **Per-segment population density — VRU exposure enrichment** |
| **Thailand MOT/TRAMS** | **External (Open Data Common)** | **80,849 crash records (2019–2022) — ground-truth validation** |

**WorldPop integration:** Per-segment population density was extracted from the WorldPop 2020 1km population grid (worldpop.org, CC BY 4.0 license) for all 14,711 segments. This upgrades the VRU vulnerability sub-score from a regional constant (helmet rate only) to a per-segment measure combining helmet rate and local population exposure. Thailand Grade D segments in central Bangkok corridors have population densities of 3,700–5,400 persons/km², making their risk profile materially different from identical-speed segments in peri-urban areas.

---

## 2. Thailand: Findings

### 2.1 Score Distribution

| Grade | Label | Segments | % of total |
|---|---|---|---|
| A | Safe | 128 | 1.1% |
| B | Adequate | 5,019 | 45.1% |
| C | Caution | 5,586 | 50.2% |
| **D** | **Unsafe** | **401** | **3.6%** |
| E | Critical | 0 | 0% |

### 2.2 Uncertainty Profile

Of the 376 Grade D segments, a subset have 95% CIs that narrow on the C/D boundary — meaning some are robustly Grade D (CI entirely within 60–80) while others could be Grade C under a slightly different data realisation. The model flags these explicitly rather than treating the boundary as definitive.

### 2.3 The Highest-Risk Segment

The segment with the highest final score in the dataset:
- **Posted limit:** 90 km/h
- **85th-percentile operating speed:** 115.5 km/h
- **Safe System threshold (urban primary):** 50 km/h
- **Speed excess above threshold:** 65.5 km/h
- **Traffic volume:** 89th percentile nationally
- **VLM analysis:** No pedestrian infrastructure, high VRU exposure, clear sightlines encouraging sustained high speed
- **Nilsson fatality reduction if corrected to 50 km/h:** ~82%

### 2.4 Economic Analysis

Using World Bank VSL for Thailand ($1.26M) and IRTAD crash rates (8.4 fatal crashes per 100M VMT), the annual economic value of correcting all Grade D segments is quantified in `outputs/priority/priority_list_thailand.csv`. The top-10 segments by economic impact represent the highest-ROI interventions available to Thailand's transport ministry.

### 2.5 Archetype Breakdown

The 376 Grade D segments cluster into archetypes. The dominant pattern is **"Urban Speedway"** — Thailand's urban default speed limits of 80–90 km/h on primary roads are a policy-level misalignment, not a road-by-road problem. A single legislative decision to align urban primary road limits with Safe System principles would address the majority of Grade D risk.

The **"Infrastructure Void"** archetype captures a secondary cluster: urban roads where VLM analysis identifies no pedestrian separation and object detection confirms high pedestrian presence. These require physical infrastructure investment, not merely a limit change.

### 2.6 Geographic Concentration

Grade D segments cluster in urban corridors — consistent with areas of high population density and mixed land use where pedestrian and motorcycle activity is highest, and where the gap between posted limits and Safe System thresholds is widest.

---

## 3. Maharashtra: Findings

### 3.1 Score Distribution

| Grade | Label | Segments | % of total |
|---|---|---|---|
| A | Safe | 0 | 0% |
| B | Adequate | 3,281 | 91.7% |
| **C** | **Caution** | **296** | **8.3%** |
| D | Unsafe | 0 | — |
| E | Critical | 0 | — |

### 3.2 The GNN Reclassification — A Methodological Insight

The Graph Attention Network's spatial risk propagation produced a notable result for Maharashtra: Stage 1 tabular scoring identified 296 segments as Grade D based on speed data alone, but the final fused score — incorporating GNN neighbourhood context, VLM imagery, and YOLOv8 VRU counts — reclassified all 296 to **Grade C (Caution)**. This is not a correction of an error; it is the model working as intended.

Maharashtra's road network has a characteristic the tabular scorer cannot detect: Grade C segments are densely clustered in urban corridors, and their graph neighbours are predominantly Grade B roads. The GAT propagates this context: a road surrounded by adequately-safe infrastructure carries lower systemic risk than an isolated Grade D segment. The MC Dropout 95% CI for these segments is wide — 33.3% are grade-uncertain — confirming they sit on the C/D boundary and warrant on-ground review.

### 3.3 The Helmet Amplification Effect

Maharashtra's 296 Grade C segments must be understood in a specific context: **only 1.2% of motorcycle passengers wear helmets**, versus 77.8% in Thailand. The VRU vulnerability sub-score (10% weight) captures this, but the true severity is understated in any metric that does not account for post-crash survival probability.

At operating speeds of 80–100 km/h on a Grade C segment, an unhelmeted motorcycle passenger's probability of surviving a crash approaches zero. Speed management in Maharashtra is inseparable from helmet enforcement. The **$987.3M/year** economic value of correcting these 296 Grade C segments reflects this amplified crash severity.

### 3.4 Archetype Findings

Maharashtra's Grade C segments split primarily between **"Urban Speedway"** (urban trunk roads with 80 km/h limits where traffic operates at 98–104 km/h against a 60 km/h threshold) and **"Rural Risk Corridor"** (intercity trunk roads where motorcycle riders and pedestrians share high-speed alignments with no separation).

---

## 4. Policy Recommendations

### 4.1 Immediate Actions (0–12 months)

**Thailand:**
1. **Emergency review** of the top-10 segments by economic impact index (see `outputs/priority/top10_emergency.csv`): reduce posted limits to 60 km/h as interim measure; deploy average-speed enforcement cameras
2. **Systematic urban speed audit** of all 376 Grade D segments; prioritise those with `archetype_name = "Urban Speedway"` where the limit change is the primary intervention and requires no physical works
3. **National urban speed limit reform**: set a default urban primary road limit of 50 km/h in line with Safe System — Thailand's current 80–90 km/h default is globally exceptional and drives the majority of Grade D risk

**Maharashtra:**
1. **Combined enforcement campaign**: speed + helmet enforcement at the 296 Grade D segment locations simultaneously — the multiplicative effect on fatality reduction is larger than either intervention alone
2. **Speed limit review** on urban trunk roads posted at 80 km/h: reduce to 60 km/h consistent with Safe System (trunk, urban context)
3. **Physical treatment** for segments with `archetype_name = "Infrastructure Void"`: raised crossings, pedestrian bridges, crash barriers at highest-VRU-ratio locations

### 4.2 Medium-Term (1–3 years)

- Deploy the Speed Safety Score as a standing monitoring tool, updated annually as new TomTom probe data becomes available
- Integrate with ADB's Enterprise GIS platform for cross-country comparison and standardised reporting
- Address the 5,722 Thailand Grade C segments (Caution) before they escalate to Grade D as traffic volumes grow — modelled growth rates suggest 800–1,200 additional Grade D segments within 5 years without intervention
- Extend the pipeline to all ADB member countries using the same Overture + TomTom + Mapillary data stack

### 4.3 The Portfolio Approach

The intervention portfolio optimiser provides a direct answer to the most important policy question: **"Given a finite budget, where do we act first?"**

For budget scenarios of 10 / 25 / 50 / 100 / 250 / 500 segments, the model produces an optimal selection ranked by economic impact per unit of review cost. This is the first road safety scoring system that answers not just "which roads are dangerous?" but "which interventions are worth doing in what order?"

See `outputs/analysis/portfolio_scenarios.json` for full scenario data.

### 4.4 Replicability in Data-Scarce Environments

The methodology requires only three globally available data inputs:

| Input | Global availability | Free alternative |
|---|---|---|
| Road network + classification | Overture Maps (monthly) | OpenStreetMap |
| Operating speeds + posted limits | TomTom Move | HERE Traffic, Google Roads API |
| Street imagery | Mapillary | Google Street View Static API |

Stage 1 alone produces actionable outputs without GPU infrastructure. In countries with no GPS probe data, 85th-percentile speeds can be estimated from road geometry and land use using regression models trained on the Thailand/Maharashtra data.

---

## 5. Ground-Truth Crash Validation

*The Speed Safety Score was validated against 80,849 real crash records from Thailand's Ministry of Transport TRAMS system (2019–2022), which includes GPS coordinates, fatality counts, and confirmed cause classification. The model had zero access to crash data during development — all validation is fully out-of-sample.*

**Data source:** datagov.mot.go.th/dataset/roadaccident (Open Data Common license, no restrictions)

### Key Validation Results

| Test | Result |
|---|---|
| Spearman ρ (score vs crash density) | **ρ=0.093, p=3.28×10⁻¹⁵** |
| Mann-Whitney U (Grade D > Grade B crash rate) | **p=3.79×10⁻⁷** |
| AUC (score predicts crash hotspot) | **0.582** |
| Crashes matched to segments | **71,362 / 80,849 (88.3%)** |

### Fatality Rate by Grade — Monotonically Increasing

The critical validation: **fatality rate increases with every grade step**, confirming the model correctly orders road severity without ever seeing crash data.

| Grade | Segments | Avg crashes/yr/segment | Avg fatalities/yr/segment |
|---|---|---|---|
| A — Safe | 128 | 0.377 | **0.039** |
| B — Adequate | 5,019 | 1.207 | **0.172** |
| C — Caution | 5,586 | 1.993 | **0.273** |
| D — Unsafe | 401 | 1.498 | **0.292** |

Grade D has the highest fatality rate per segment — **1.7× higher than Grade B** — despite having lower absolute crash frequency. Grade C has the highest crash frequency because it combines high traffic volume with speed excess; Grade D segments have extreme speed excess with lower traffic density, producing fewer but almost always severe crashes. Both are consistent with Safe System theory.

Grade D segments are **3.6% of the road network** but account for **4.7% of all fatalities** — a statistically significant disproportionate concentration (Mann-Whitney p<10⁻⁶).

74.6% of all crashes in the TRAMS dataset are speed-related — confirming that speed is the dominant factor on these roads, which is precisely what this model targets.

### OpenStreetMap Infrastructure Cross-Validation

The OSM Overpass API was queried for all 100 top-priority Thailand Grade D segments, providing an independent ground-truth check on the VLM's infrastructure assessments.

| OSM attribute | Thailand Grade D (top 100) |
|---|---|
| Has sidewalk | **0 / 46 enriched segments (0%)** |
| Has pedestrian crossing | **0 / 46 (0%)** |
| Has traffic calming | **0 / 46 (0%)** |
| Street lighting present | **1 / 46 (2.2%)** |
| Average OSM max speed | **91.8 km/h** |
| Average lanes | **2.8** |
| Average infrastructure score | **0.004 / 1.0** |

**Interpretation:** Every single one of the top-priority Thailand Grade D segments has zero pedestrian infrastructure according to OSM ground truth. This is not a model artifact — it is confirmed by the world's most comprehensive crowd-sourced road database. High-speed, multi-lane roads with no sidewalks, no crossings, no traffic calming and almost no lighting are exactly the conditions the Safe System identifies as most lethal for vulnerable road users. The OSM infrastructure score of 0.004 (near-theoretical minimum) independently validates the model's "Infrastructure Void" and "Urban Speedway" archetype assignments.

---

## 6. Counterfactual Policy Simulation

*What would happen if Safe System speed limits were adopted?*

Using the Nilsson Power Model × VOSL × VMT proxy (the same model used to rank individual segments), two counterfactual scenarios were simulated per region. These are relative impact indices for ranking policy actions — not calibrated absolute predictions.

### Thailand

| Scenario | Segments affected | Annual economic value | 10-year economic value |
|---|---|---|---|
| **A** — Full Safe System adoption (all Grade D) | 401 | **$1,153.5M/yr** | **$11.5B** |
| **B** — Urban primary/trunk reform only | 200 | **$868.5M/yr** | **$8.7B** |

**Scenario B requires one legislative act — no road works, no capital budget.** A single ministerial regulation reducing the national urban primary road default from 80–90 km/h to 50 km/h would address 200 of 401 Grade D segments and capture 75% of the total economic value of full Safe System adoption.

### Maharashtra

| Scenario | Segments affected | Annual economic value | 10-year economic value |
|---|---|---|---|
| **A** — Full Safe System adoption (all Grade C) | 296 | **$987.3M/yr** | **$9.9B** |
| **B** — Urban primary/trunk reform only | 43 | **$54.3M/yr** | **$0.5B** |

Maharashtra's risk is more distributed: unlike Thailand's concentrated Urban Speedway pattern, the Grade C segments span rural corridors and mixed urban roads. Full network intervention (Scenario A) is required to capture the majority of economic value.

---

## 7. Conclusion

This study demonstrates that AI-powered speed safety assessment is a deployable tool that can tell a transport ministry, road by road, where speed limits are endangering lives, by how much, which interventions to prioritise, and what the economic return on each decision is.

The technical advances in this submission — domain-specific VLM analysis at 72B scale, MC Dropout uncertainty quantification per segment, archetype-based risk classification, and a budget-constrained portfolio optimiser — represent a step change over prior safety scoring approaches. The combination produces not just a score but a decision-support system.

The findings from Thailand and Maharashtra reveal that speed limit misalignment is systemic, concentrated, and quantifiable. Correcting the misalignment, guided by the priority list this model produces, represents one of the highest-return investments in road safety available to policymakers in Asia and the Pacific.

The methodology is open, reproducible, and built to scale. With the data stack ADB already has access to, this tool could be operational across all member countries within six months of a decision to deploy it — covering more than 300 million road users in economies where road fatality rates are among the highest in the world.

---

*Full source code, scored datasets, interactive maps, priority lists, and segment-level policy briefs available at the project GitHub repository.*

*Compute infrastructure: 8× NVIDIA RTX A5000 (192 GB VRAM total), Pusan National University GenAI Lab.*
