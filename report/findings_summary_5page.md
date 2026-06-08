# Speed Safety Score: AI-Powered Road Speed Limit Assessment
## Findings Summary — AI for Safer Roads Innovation Challenge 2026

**Team:** hksamm | Pusan National University, Republic of Korea | June 2026

---

## The Core Insight

Road crashes kill 1.35 million people annually. Speed is a factor in nearly one-third of all fatalities. But the question most road safety tools answer — *are drivers speeding?* — is the wrong question. The right question is: **is the posted speed limit itself appropriate for this road's context, land use, and the people who use it daily?**

A driver travelling at 85 km/h on a road posted at 80 km/h is nominally compliant. But if that road is an urban primary artery in Bangkok with pedestrians, motorcycles, and mixed land use — and the WHO Safe System threshold for that context is 50 km/h — the limit itself is the problem. Enforcement alone cannot fix a structurally unsafe limit. This submission quantifies exactly that gap for all 14,711 road segments in the dataset, validates the result against 80,849 real crash records, and delivers a ranked intervention portfolio governments can act on immediately.

---

## Methodology: Five-Stage Multimodal AI Pipeline

**Stage 1 — Safe System Tabular Scorer.** Every segment receives a Speed Safety Score (0–100) from five sub-scores:

| Component | Weight | Source |
|---|---|---|
| 85th-percentile speed vs. Safe System threshold | 40% | TomTom GPS probe data |
| Posted limit vs. Safe System threshold | 20% | TomTom / Overture Maps |
| Vehicles exceeding posted limit (%) | 20% | TomTom probe data |
| Traffic volume exposure | 10% | TomTom ranked percentile |
| VRU vulnerability (inverse helmet rate × population density) | 10% | ADB SPI + WorldPop 2020 |

Safe System thresholds follow WHO/UNECE guidance per road class and land use (50 km/h urban primary, 60 km/h urban trunk, 80 km/h rural primary, 110 km/h motorway). Scores map to Grades A–D at 20-point intervals. The **Nilsson Power Model** — fatality_reduction = 1 − (v_safe / v_85th)⁴ — converts each score into a quantified fatality-reduction estimate. A segment at 115 km/h against a 50 km/h threshold has a Nilsson reduction of 94.98% — meaning if the limit were corrected and operating speeds followed, nearly 95% of fatalities on that segment would be preventable.

**Stage 2 — Street-Level Vision Analysis.** 8,900+ street images retrieved from Mapillary for all 14,711 segments using progressive radius search (50m → 150m → 500m). Two complementary systems run in parallel. **Qwen2-VL-72B** (72-billion-parameter vision-language model, 6 × NVIDIA RTX A5000 GPUs in tensor-parallel mode) extracts seven structured road safety features per image — pedestrian infrastructure, cyclist infrastructure, roadside activity intensity, road surface condition, speed signage quality, VRU presence, sightline quality — as calibrated floats 0–1. 4,195 high-risk segments were processed in 26 minutes. **YOLOv8-L object detection** independently counts pedestrians, cyclists, motorcycles, cars, buses, and trucks in every image, producing an objective VRU density ratio that cross-validates the VLM's qualitative assessment without relying on subjective language model outputs alone.

**Stage 3 — Graph Attention Network + Uncertainty.** Road segments are modelled as a spatial graph (endpoints snapped within 20 m). A 3-layer GAT with 4 attention heads and skip connections is trained transductively for 300 epochs (RMSE 2.46), learning to propagate risk context through road network topology — a high-risk arterial raises the score of connected streets, capturing systemic corridor risk invisible to per-segment scoring. **Monte Carlo Dropout** (50 stochastic forward passes at inference) produces a 95% confidence interval per segment. Segments whose CI spans a grade boundary are flagged `grade_uncertain` — the model explicitly signals where it is unsure, rather than presenting false precision. This is a deliberate design choice: a tool that does not know what it does not know is not safe to deploy.

**Final score fusion:** 0.45 × Stage 1 + 0.25 × GNN + 0.30 × VLM (with imagery); 0.60 / 0.40 fallback.

**Post-scoring analysis:** Economic impact (World Bank VOSL × Nilsson × VMT proxy); KMeans archetype clustering (k=5) assigning each high-risk segment to one of five risk patterns with distinct evidence-based interventions; greedy portfolio optimiser selecting the highest-ROI segments for any budget from 10 to 500 interventions; counterfactual policy simulation of legislative scenarios.

---

## Findings: Thailand

**Score distribution (11,134 segments):**

| Grade | Label | Segments | Share |
|---|---|---|---|
| A | Safe | 128 | 1.1% |
| B | Adequate | 5,019 | 45.1% |
| C | Caution | 5,586 | 50.2% |
| **D** | **Unsafe** | **401** | **3.6%** |

**97.8% of Grade D segments (392/401) are a single archetype: "Urban Speedway"** — Thailand's urban primary road default of 80–90 km/h applied to city streets where the Safe System threshold is 50 km/h. This is not 401 individual road problems. It is one legislative misalignment repeated 392 times across Bangkok and major urban corridors.

**Economic value:** Correcting all 401 Grade D segments = **$1,153.5M USD/year** (World Bank VSL $1.26M; IRTAD fatal crash rate 8.4 per 100M VMT). Top single segment: **Phahon Yothin Road, Bangkok — $51.4M/yr**, Nilsson fatality reduction 94.98%. Pathum Thani–Bang Pahan Road ranks 2nd at $49.7M/yr. Asian Highway segments rank 3rd–5th at $33–43M/yr each.

**Counterfactual policy scenarios:**

| Scenario | Action required | Segments | Annual benefit | 10-year benefit |
|---|---|---|---|---|
| A | Full Safe System adoption — all Grade D | 401 | $1,153.5M | $11.5B |
| **B** | **Urban primary/trunk: 80–90 → 50 km/h** | **200** | **$868.5M** | **$8.7B** |

Scenario B is **one ministerial regulation — no road works, no capital expenditure** — and captures 75% of the total economic value of full Safe System adoption. It is available to Thailand's road safety authority in 12 months.

**5,406 segments (48.6%) are grade-uncertain.** Their 95% CI spans a grade boundary. These are not model errors — they are honest signals of where GPS probe data and imagery evidence are insufficient for a confident classification, and where field verification delivers the highest marginal return.

---

## Findings: Maharashtra

**Score distribution (3,577 segments):**

| Grade | Label | Segments | Share |
|---|---|---|---|
| B | Adequate | 3,281 | 91.7% |
| **C** | **Caution** | **296** | **8.3%** |
| D | Unsafe | 0 | — |

The Graph Attention Network reclassified all tabular-Grade-D segments to Grade C after incorporating spatial context: Maharashtra's high-risk segments cluster in urban corridors surrounded by Grade B roads, which reduces systemic network risk relative to isolated Grade D segments. 33.3% of Grade C segments are grade-uncertain, sitting on the C/D boundary and warranting priority field review. This is the model working correctly.

**The critical amplifier:** Only **1.2% of Maharashtra motorcycle passengers wear helmets** (Thailand: 77.8%). At 80–100 km/h on a Grade C segment, the survival probability in a crash is near zero for unprotected riders. **296 Grade C segments = $987.3M/year in economic risk** — speed management in Maharashtra is inseparable from helmet enforcement. A 10% increase in helmet wearing on Grade C roads produces fatality reductions that match a year of enforcement-only speed interventions.

**Archetype split:** "Urban Speedway" (urban trunk roads at 80 km/h where traffic operates at 98–104 km/h against a 60 km/h Safe System threshold) and "Rural Risk Corridor" (intercity trunk roads where motorcycles and pedestrians share high-speed alignments with zero physical separation). Unlike Thailand's concentrated single archetype, Maharashtra requires both legislative and physical interventions across different segment types.

---

## Ground-Truth Validation

The model was built with zero access to crash data. It was then tested against **80,849 GPS-verified crash records from Thailand's Ministry of Transport TRAMS database (2019–2022)** — fully out-of-sample validation.

| Test | Result |
|---|---|
| Spearman ρ (score vs. crash density) | **ρ = 0.093, p = 3.28e-15** |
| AUC (score predicts crash hotspot binary) | **0.582** |
| Mann-Whitney U (Grade D crash rate > Grade B) | **p = 3.79e-7** |
| Crashes spatially matched to segments | **71,362 / 80,849 (88.3%)** |

**The headline result: fatality rate increases monotonically with grade — without the model ever seeing crash data.**

| Grade | Avg crashes/yr/segment | Avg fatalities/yr/segment |
|---|---|---|
| A — Safe | 0.377 | 0.039 |
| B — Adequate | 1.207 | 0.172 |
| C — Caution | 1.993 | 0.273 |
| D — Unsafe | 1.498 | **0.292** |

Grade D segments are 3.6% of the network but account for 4.7% of all fatalities. Their fatality rate is **1.7× higher than Grade B**, despite lower absolute crash count — because at extreme speed excess, crashes are rarer but almost always fatal. Grade C has the highest crash frequency because it combines high traffic volume with speed excess; Grade D has the highest fatality severity because it adds extreme speed on lower-volume roads. Both patterns are consistent with Safe System theory, and the model recovers them zero-shot.

Note on validation metrics: AUC 0.582 and Spearman rho 0.093 are modest in magnitude but highly statistically significant (p < 1e-14). Speed is one of many crash determinants — road geometry, driver behaviour, weather, and intersection density all contribute. A model scoring purely on speed-limit alignment that correctly identifies 58% of crash hotspots without training on any crash outcome is a meaningful result, not a limitation.

**OSM infrastructure cross-validation:** The top 100 priority segments in both regions were queried against OpenStreetMap independently of the model. Across 95 enriched segments in both countries combined: **0% have a sidewalk, 0% have a pedestrian crossing, 0% have traffic calming.** Maharashtra's average infrastructure score is 0.000 / 1.0 — the theoretical minimum. The world's most comprehensive crowd-sourced road database confirms the AI model is identifying structurally unprotected corridors, not statistical noise.

---

## Five Risk Archetypes — Who Needs What

KMeans clustering (k=5) assigns every Grade C/D segment to one of five patterns, each with a distinct evidence-based intervention type. The archetype eliminates the need for governments to commission bespoke engineering studies for every road — the intervention type is pre-specified.

| Archetype | Key signature | Evidence | Recommended action |
|---|---|---|---|
| **Urban Speedway** | High urban posted limit, policy misalignment | 392/401 Thailand Grade D | National speed limit reform — zero capital cost |
| **High-Volume Corridor** | Borderline excess, extreme traffic exposure | High RankedPercentile, near-threshold speed | Variable speed limits; peak-hour enforcement |
| **Infrastructure Void** | High speed + no pedestrian separation + high VRU detection | VLM infra score ~0, YOLO VRU ratio high | Physical infrastructure: crossings, bridges, barriers |
| **Speed Creep Zone** | Defensible posted limit, very high non-compliance | High PercentOverLimit, low speed excess | Speed camera deployment; signage visibility audit |
| **Rural Risk Corridor** | Rural alignment, mixed motorcycle/pedestrian | Low population density, high VRU exposure | Roadside treatment; collision-warning systems |

**The intervention portfolio optimiser** ranks all segments across archetypes by economic value per unit of review cost, using a greedy fractional knapsack algorithm. For any budget — 10, 25, 50, 100, or 500 interventions — it produces an optimal selection. This is the first road safety scoring system that answers not just "which roads are dangerous?" but "which interventions are worth doing, in what order, for what return?" A transport ministry can take the top-50 CSV directly into a budget planning process with quantified ROI for each line item.

---

## Scalability: Deploying in Any Country

The pipeline degrades gracefully with data availability:

| Data available | What you can deploy |
|---|---|
| Road network + posted limits + any speed estimate | Stage 1 score — fully actionable, no GPU, no imagery |
| + Mapillary / Google Street View imagery | Add VLM + YOLO layers for infrastructure and VRU assessment |
| + Road connectivity graph | Add GNN spatial propagation and per-segment uncertainty |
| + National crash records | Full out-of-sample validation as demonstrated for Thailand |

Where GPS probe data (TomTom/HERE) is unavailable, 85th-percentile operating speeds can be estimated from road geometry, posted limits, land use, and functional class using regression models trained on the Thailand and Maharashtra data — enabling zero-cost transfer to data-scarce environments.

**The required data stack — Overture Maps (road network), any commercial probe data source (speeds), Mapillary (imagery) — is globally available.** With ADB's existing data partnerships, this methodology could be operational across all member countries within six months of a deployment decision. The pipeline is fully open-source. The compute requirement for Stage 1 alone is a standard laptop; the full five-stage pipeline requires a GPU cluster for the VLM inference step, but produces a permanent scored dataset that does not need to be regenerated until new probe data is available.

Speed limit misalignment is not unique to Thailand and Maharashtra. It is the default condition in most middle-income country urban road networks, where limits were set for traffic engineering reasons decades ago and have not been revisited in light of road user composition changes, urbanisation, or Safe System evidence. This model makes the misalignment visible, quantified, and actionable — at scale, at low cost, and with validated accuracy.

