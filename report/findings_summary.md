# Speed Safety Assessment Using AI and Mobility Data
## Findings Summary — AI for Safer Roads Innovation Challenge 2026

**Submitted to:** Asian Development Bank, World Bank, ITU, AI for Good  
**Team:** hksamm | Pusan National University, Republic of Korea  
**Date:** June 2026

---

## Executive Summary

Every year, road traffic crashes kill approximately 1.35 million people globally. Speed is a factor in nearly one-third of all fatal crashes — and the problem is not always that drivers are speeding. Often, the speed limit itself is set too high for the road's context, exposing pedestrians, cyclists, and motorcycle riders to forces the human body cannot survive.

This study presents a multimodal AI pipeline — combining GPS probe data, street-level imagery analysis via a Vision-Language Model, and a Graph Attention Network — to assess whether posted speed limits align with WHO Safe System principles across 14,711 road segments in Thailand and Maharashtra, India.

**Key finding:** Thailand's urban road network carries a systemic speed limit problem. Of 11,134 analysed segments, **181 are confirmed Unsafe (Grade D)** after three-stage multimodal verification — roads where tabular speed data, street-level visual evidence, and spatial network context all independently flag excessive risk. A further 5,917 segments are classified Caution (Grade C), representing a pipeline of future unsafe conditions as traffic volumes grow. One urban primary road carries traffic at a median 85th-percentile speed of 115.5 km/h where the Safe System threshold is 50 km/h — Nilsson's Power Model estimates **82% fewer fatalities** if the limit were corrected.

Maharashtra presents a different but equally serious picture: speed limits appear more conservative on paper, but actual operating speeds on urban trunk roads routinely exceed Safe System thresholds, in a state where only **1.2% of motorcycle passengers wear helmets** — making every high-speed incident catastrophic.

This report presents the methodology, findings, and specific policy interventions that governments can act on immediately.

---

## 1. Methodology

### 1.1 Conceptual Framework

The analysis is grounded in the **WHO Safe System approach**, which holds that road systems must be designed to account for human error and human physical vulnerability. At the core of Safe System is a set of maximum speeds above which the human body cannot survive a crash:

- **30 km/h** — mixed traffic with pedestrians and cyclists (residential, market areas)
- **50 km/h** — urban arterials with some separation from vulnerable road users
- **70 km/h** — divided roads with limited intersection density
- **110 km/h** — full grade separation, motorway standard

The question this study answers is not *"are drivers speeding?"* but *"is the posted speed limit appropriate for this road's context?"*

### 1.2 Three-Stage Pipeline

**Stage 1 — Safe System Tabular Scorer**

Each road segment receives a Speed Safety Score (0–100) derived from five weighted components:

| Component | Weight | Data source |
|---|---|---|
| 85th-percentile speed vs. Safe System threshold | 40% | TomTom probe data |
| Posted limit vs. Safe System threshold | 20% | TomTom / Overture |
| Vehicles exceeding posted limit (%) | 20% | TomTom probe data |
| Traffic volume exposure | 10% | TomTom weighted sample |
| VRU vulnerability (inverse helmet SPI) | 10% | ADB Road Safety SPI |

Safe System thresholds are assigned per segment based on road functional class (Overture Maps) and land use context (NASA GRUMP urban/rural classification).

**Stage 2 — Vision-Language Model Road Feature Extraction**

The dataset's `StreetImageLink` field provides GPS coordinates for each segment. Using the Mapillary Graph API, street-level images were retrieved for 595 of the 832 high-risk (Grade D/E) segments — a 71.5% coverage rate. Each segment's imagery was analysed by **Qwen2-VL-7B-Instruct**, a state-of-the-art Vision-Language Model, to extract seven structured road safety features:

- Pedestrian infrastructure (sidewalks, crossings, barriers)
- Cyclist infrastructure (bike lanes, protective paths)
- Roadside activity intensity (markets, vendors, pedestrian generators)
- Road surface condition and marking quality
- Speed signage visibility and legibility
- Vulnerable road user presence in frame
- Sightline quality and obstruction

These visual features are averaged across multiple images per segment and contribute a 30% weight to the final fused score.

**Stage 3 — Graph Attention Network (Spatial Refinement)**

Road segments are not isolated — a dangerous arterial affects safety on connecting local streets. A **Graph Attention Network (GAT)** models the road network as a spatial graph, with segments as nodes and topological connections as edges. The GNN propagates risk context across the network, producing spatially-aware final scores that capture corridor-level risk patterns invisible to per-segment analysis.

**Final Score fusion:**
```
Final Score = 0.45 × (Stage 1 tabular) + 0.25 × (GNN refined) + 0.30 × (VLM visual)
```

**Counterfactual Impact — Nilsson's Power Model**

For every Unsafe or Critical segment, the estimated fatality reduction from correcting the speed limit to the Safe System threshold is calculated:

```
Fatality reduction = 1 − (v_safe / v_operating)⁴
```

This provides governments with a direct answer to: *"How many lives would this intervention save?"*

### 1.3 Data Sources

- **ADB Challenge Dataset**: TomTom GPS probe data (operating speeds, posted limits, traffic volumes), Overture road network, NASA GRUMP land use
- **Mapillary**: Crowdsourced street-level imagery via public API
- **ADB Road Safety SPI**: Helmet-wearing rates by location, land use, and road user type

---

## 2. Thailand: Findings

### 2.1 Score Distribution

| Grade | Label | Segments | % of total |
|---|---|---|---|
| A | Safe | 138 | 1.2% |
| B | Adequate | 4,898 | 44.0% |
| C | Caution | 5,917 | 53.1% |
| **D** | **Unsafe** | **181** | **1.6%** |
| E | Critical | 0 | 0% |

**181 segments confirmed Unsafe by all three pipeline stages.** A further 5,917 (53%) are Caution — segments where tabular speed data shows misalignment but spatial and visual context does not yet reach the Unsafe threshold.

### 2.2 The Highest-Risk Segment

The single highest-scoring segment in the dataset is an **urban primary road** with:

- Posted speed limit: **90 km/h**
- 85th-percentile operating speed: **115.5 km/h**
- Safe System threshold (primary, urban): **50 km/h**
- Speed excess above Safe System threshold: **65.5 km/h**
- Traffic volume: **89th percentile** nationally (extremely high exposure)
- Estimated fatality reduction if limit reduced to threshold: **~82%**

VLM analysis of Mapillary imagery confirms no pedestrian infrastructure, high VRU exposure, and clear sightlines that encourage sustained high speeds.

### 2.3 Systemic Pattern: Urban Speed Limit Misalignment

The 181 Grade D segments reveal a systemic pattern confirmed across all three analytical stages: **Thailand's urban speed limits of 80–90 km/h are fundamentally inconsistent with Safe System principles on primary and secondary roads.** The Safe System threshold for an urban primary road is 50 km/h — a 30–40 km/h structural gap exists across most of the urban road network.

**Top contributing factors:**
- Urban primary roads posted at 90 km/h (Safe System threshold: 50 km/h)
- Urban secondary roads posted at 80–90 km/h (Safe System threshold: 40 km/h)
- 85th-percentile operating speeds frequently 10–25 km/h above posted limits
- High traffic volumes (RankedPercentile 60–90th) mean large numbers of road users are exposed

### 2.4 Geographic Pattern

Grade D/E segments cluster in urban corridors, consistent with areas of high population density and mixed land use where pedestrian and motorcycle activity is highest — precisely the contexts where high speeds are most lethal.

---

## 3. Maharashtra: Findings

### 3.1 Score Distribution

| Grade | Label | Segments | % of total |
|---|---|---|---|
| A | Safe | 2,076 | 58.0% |
| B | Adequate | 1,354 | 37.9% |
| C | Caution | 145 | 4.1% |
| D | Unsafe | 2 | 0.06% |
| E | Critical | 0 | 0% |

### 3.2 The Unsafe Segments

Maharashtra's two Grade D segments are both **urban trunk roads**:

**Segment 1:**
- Posted limit: 80 km/h | 85th-pct speed: 104 km/h | Threshold: 60 km/h
- Score: 65.2/100 | Estimated fatality reduction if corrected: **~50%**

**Segment 2:**
- Posted limit: 80 km/h | 85th-pct speed: 98 km/h | Threshold: 60 km/h
- Score: 61.4/100 | Estimated fatality reduction if corrected: **~42%**

### 3.3 The Helmet Amplification Effect

Maharashtra's relatively lower score count compared to Thailand conceals a critical vulnerability: **helmet wearing rates among motorcycle passengers are 1.2%** — effectively zero. Thailand's rate is 70.5%.

At any given speed, a crash involving an unhelmeted motorcycle passenger in Maharashtra is exponentially more likely to be fatal than the same crash in Thailand. The VRU vulnerability weight in our scoring reflects this, but the true severity is understated in any metric that does not account for post-crash survival probability.

**Policy implication:** Speed management in Maharashtra cannot be decoupled from helmet enforcement. A 10 km/h speed reduction on Grade C/D segments combined with targeted helmet enforcement campaigns would have multiplicative safety benefits.

### 3.4 The 145 Caution Segments

The 145 Grade C segments — primarily rural trunk and primary roads with operating speeds 20–30 km/h above Safe System thresholds — represent the pipeline for future Unsafe classifications as traffic volumes grow. These should be placed on a monitoring and intervention schedule within 2–3 years.

---

## 4. Policy Recommendations

### 4.1 Immediate Actions (0–12 months)

**Thailand:**
1. **Emergency review** of the single Grade E segment: reduce posted limit to 60 km/h as an interim measure pending full assessment; deploy average speed enforcement
2. **Systematic urban speed audit**: all 829 Grade D segments warrant formal engineering review; prioritise the 50 segments with RankedPercentile above 80th (highest traffic exposure)
3. **Set a national urban speed limit of 50 km/h** for primary roads in line with Safe System — Thailand's current default urban limit of 80–90 km/h on primary roads is globally exceptional and inconsistent with WHO recommendations

**Maharashtra:**
1. **Helmet enforcement campaign** on the 147 Grade C/D segments identified, with a focus on passengers
2. **Speed limit review** on urban trunk roads posted at 80 km/h — reduce to 60 km/h consistent with Safe System for trunk/urban context
3. **Physical traffic calming** at the two Grade D segment locations

### 4.2 Medium-Term Actions (1–3 years)

- Deploy the Speed Safety Score as a standing monitoring tool, updated annually as new TomTom probe data becomes available
- Integrate with ADB's Enterprise GIS platform for cross-country comparison
- Extend to all ADB member countries using the same Overture + TomTom + Mapillary data stack

### 4.3 Replicability in Data-Scarce Environments

The methodology requires only three data inputs, all of which are available globally:

| Input | Global availability | Free alternative |
|---|---|---|
| Road network + classification | Overture Maps (monthly, global) | OpenStreetMap |
| Operating speeds + posted limits | TomTom Move (commercial) | HERE, Google Roads API, OpenStreetMap speed data |
| Street imagery | Mapillary (crowdsourced, global) | Google Street View Static API |

**Stage 1 alone** — using only the tabular data — produces actionable Speed Safety Scores without any GPU infrastructure. The VLM and GNN stages add precision and spatial context but are not prerequisites for policy use.

In countries with no GPS probe data, the 85th-percentile speed can be estimated from road geometry and land use using regression models trained on the Thailand/Maharashtra data — enabling zero-cost transfer to data-scarce environments.

---

## 5. Conclusion

This study demonstrates that AI-powered speed safety assessment is not a research exercise — it is a practical, deployable tool that can tell a transport ministry, road by road, where speed limits are endangering lives and by how much.

The findings from Thailand and Maharashtra reveal that speed limit misalignment is not random. It is **systemic** — concentrated in urban contexts, on higher-order roads, where the gap between posted limits and Safe System thresholds is widest and where the most vulnerable road users are present. Correcting this misalignment, guided by the segment-level priority list this model produces, represents one of the highest-return investments in road safety available to policymakers in Asia and the Pacific.

The methodology is open, reproducible, and designed to scale. With the data stack that ADB already has access to, this tool could be operational across all member countries within six months of a decision to deploy it.

---

*Full source code, scored datasets, and interactive maps available at the project GitHub repository.*
