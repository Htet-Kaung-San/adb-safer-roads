# ADB Challenge Submission Form — Text for All Fields
## Ready to copy-paste into the ADB Challenges platform

---

## PARTICIPATION NAME
```
hksamm — Speed Safety Score: AI-Powered Road Speed Limit Assessment
```

## DESCRIPTION (252 chars max)
```
A three-stage AI pipeline combining GPS probe data, Vision-Language Model street imagery analysis, and Graph Neural Networks to identify road segments where speed limits endanger lives — with counterfactual impact estimates per segment.
```

---

## SECTION 1: TEAM INFORMATION

**Team Name:**
```
hksamm
```

**Team Lead — Complete Name:**
```
Htet Kaung San
```

**Email:**
```
htet_kaung_san@pusan.ac.kr
```

**Country of Residence:**
```
Republic of Korea
```

**Organization / Institution:**
```
Pusan National University
```

**Highest Educational Attainment:**
```
Bachelor's Degree (currently enrolled)
```

**Team Members:**
```
Htet Kaung San
Year of birth: [your birth year]
Country of Residence: Republic of Korea
Email: htet_kaung_san@pusan.ac.kr
```

**Team Composition:**
```
This is a solo submission. I am a full-stack software engineer and AI engineer with hands-on experience building production machine learning systems. My technical stack spans Python, geospatial data processing (GeoPandas, Shapely), deep learning (PyTorch, HuggingFace Transformers), graph neural networks (PyTorch Geometric), and large language/vision models. I have access to a GPU computing cluster (8× NVIDIA RTX A5000, 192GB total VRAM) which enables the multimodal analysis at the core of this submission. My background in both software engineering and AI allows me to build end-to-end systems — from raw data ingestion to policy-ready geospatial visualizations — without dependency on a larger team.
```

**Previous Experience:**
```
I have practical experience building AI systems across computer vision, natural language processing, and geospatial analysis domains. Relevant experience for this challenge includes:

- Building deep learning pipelines for image classification and object detection using PyTorch and HuggingFace
- Geospatial data processing and analysis using GeoPandas, Shapely, and Folium for map-based visualizations
- Working with large-scale GPS and mobility datasets for pattern analysis
- Implementing and fine-tuning large vision-language models (LLaVA, Qwen-VL) for structured information extraction from images
- Graph neural network implementation for spatial and network data using PyTorch Geometric
- Full-stack development experience enabling deployment of AI models into interactive web-based dashboards

This challenge brings together all of these domains in a way that is directly relevant to my academic and professional development.
```

---

## CHALLENGE SUBMISSION

**Submission Title:**
```
Speed Safety Score: A Multimodal AI Pipeline for Road Speed Limit Assessment Using GPS Probe Data, Street Imagery, and Graph Neural Networks
```

---

**Executive Summary** *(for a non-technical audience — transport ministry official)*
```
Every year, road crashes kill more than 1.35 million people globally. Speed is a contributing factor in nearly one third of all fatal crashes. But the problem is not always that drivers are going too fast for the posted limit. Sometimes, the limit itself is wrong — set too high for the type of road, the surrounding environment, and the people who use it. A 90 km/h limit on an urban road where motorcyclists and pedestrians mix daily is not a safe limit, regardless of whether drivers are technically obeying it.

This submission presents a tool that answers a straightforward question for every road segment in the dataset: Is this speed limit safe for the people who use this road?

We built a three-stage AI system. First, we score each road segment using GPS probe data — comparing actual operating speeds against internationally established Safe System thresholds, accounting for road type, urban or rural context, and how much traffic the road carries. Second, we use a Vision-Language Model — the same type of AI that powers advanced image understanding systems — to analyze street-level photographs of each road segment, detecting whether there are sidewalks, whether markets or schools are nearby, whether cyclists and pedestrians are visible, and whether speed signs are legible. Third, we apply a Graph Neural Network that understands road network connections, so that a dangerous arterial road raises the risk score of the local streets connected to it.

The result is a Speed Safety Score for every segment, classified from Grade A (Safe) to Grade E (Critical), accompanied by an estimate of how many fewer fatalities there would be if the speed limit were corrected.

Our key finding: Thailand has a systemic problem. Of 11,134 road segments analyzed, 830 are classified Unsafe or Critical — roads where posted limits of 80–90 km/h in urban areas are 30 to 65 km/h above what the World Health Organization's Safe System framework recommends. One segment carries traffic at a median speed of 115 km/h in an urban area where the safe limit should be 50 km/h. Our model estimates that correcting that single segment alone would reduce fatalities on it by 82%.

In Maharashtra, the picture is different but equally serious: speed limits appear more conservative on paper, but only 1.2% of motorcycle passengers wear helmets. At these speeds, that means every serious crash is likely fatal.

This tool is designed to be used by transport ministries and road safety agencies to prioritize which roads need attention first, what limit changes would have the greatest impact, and how to make the case for intervention with quantified evidence. It is built to scale to any country with access to GPS probe data and road network information — which today means almost anywhere in the world.
```

---

**Methodology Description**
```
DATA SOURCES

The analysis uses the ADB-provided dataset comprising:
- TomTom GPS probe data: operating speeds (median and 85th-percentile), posted speed limits, percentage of vehicles exceeding the posted limit, and traffic volume indicators (weighted sample size, ranked percentile)
- Overture Maps road network: functional classification (motorway, trunk, primary, secondary) and segment geometry
- NASA GRUMP: urban/rural land use classification per segment
- ADB Road Safety Performance Indicators: helmet-wearing rates by location, land use, and road user type (driver vs. passenger)
- Mapillary street-level imagery: accessed via the public Mapillary Graph API using GPS coordinates provided in the StreetImageLink field

DATA PREPROCESSING

Segments were filtered to those with valid TomTom data (AnalysisStatus = 'Valid', SampleSize > 0), yielding 3,577 usable segments for Maharashtra and 11,134 for Thailand. Column naming inconsistencies between the two datasets were resolved programmatically. RankedPercentile values were normalized within each dataset (Maharashtra uses a 0–1 scale; Thailand uses 0–100) to enable consistent cross-dataset scoring.

MODEL ARCHITECTURE

Stage 1 — Safe System Tabular Scorer

Safe System speed thresholds are assigned per segment based on road functional class and land use context, following WHO Safe System principles and UNECE road safety guidelines:
- Motorway (urban/rural): 80/110 km/h
- Trunk (urban/rural): 60/80 km/h
- Primary (urban/rural): 50/80 km/h
- Secondary (urban/rural): 40/60 km/h

Five sub-scores are computed and combined with fixed weights:
(1) Speed deviation (40%): how far the 85th-percentile operating speed exceeds the Safe System threshold, normalized to [0,1] with a 60 km/h cap
(2) Posted limit excess (20%): how far the posted limit itself exceeds the Safe System threshold, normalized with a 50 km/h cap
(3) Speeding prevalence (20%): percentage of vehicles exceeding the posted limit (PercentOverLimit, already 0–1)
(4) Traffic exposure (10%): traffic volume proxy using within-dataset normalized RankedPercentile
(5) VRU vulnerability (10%): inverse of the regional helmet-wearing rate, normalized against a 90% baseline — reflecting that crashes at identical speeds are more fatal where fewer road users are protected

Stage 2 — VLM Street Imagery Analysis

For Grade D and E segments, street-level images are retrieved from the Mapillary API using a progressively expanding search radius (50m → 150m → 500m) to maximize coverage. Images are analyzed by Qwen2-VL-7B-Instruct, a Vision-Language Model running on an 8-GPU NVIDIA RTX A5000 cluster. The model is prompted to return structured JSON scores (0–1) for seven road safety features: pedestrian infrastructure, cyclist infrastructure, roadside activity intensity, road surface condition, signage quality, VRU presence, and sightline quality. Results are averaged across up to three images per segment.

Stage 3 — Graph Attention Network

The road network is modeled as a spatial graph by snapping segment endpoints within 20 metres to create edges. A 3-layer Graph Attention Network (GAT) with 4 attention heads and skip connections processes the full feature matrix (Stage 1 sub-scores + VLM features + road class and land use one-hot encodings). The GNN is trained in a transductive setting using Stage 1 scores as the learning target, learning to propagate risk context across network topology. This ensures that a high-risk arterial elevates scores on connected segments even if those segments individually appear borderline.

FINAL SCORE FUSION
Final Score = 0.45 × (Stage 1) + 0.25 × (GNN) + 0.30 × (VLM features mean × 100)
Segments without VLM imagery fall back to: 0.60 × (Stage 1) + 0.40 × (GNN)

COUNTERFACTUAL IMPACT

Nilsson's Power Model is applied to every Unsafe or Critical segment to estimate the fatality reduction achievable by reducing the operating speed to the Safe System threshold:
Fatality reduction = 1 − (v_safe / v_85th)⁴

VALIDATION APPROACH AND LIMITATIONS

Internal validation: Score rankings were cross-checked against known road safety risk factors (road class, land use, traffic volume) to confirm face validity. Segments with the highest scores consistently show the expected combination of high-class urban roads, high traffic volumes, and large speed-threshold gaps.

Limitations: (1) No ground-truth crash data was available for quantitative validation; absolute fatality estimates are relative indices, not confirmed counts. (2) Speed limit data is TomTom-derived and not independently validated — limits that are listed as identical across a long section may vary on the ground. (3) Mapillary coverage is uneven; 28.5% of high-risk segments had no retrievable imagery and fell back to tabular scoring only. (4) The GNN is trained on the same data it is evaluated on (transductive setting) — this is appropriate for a spatial smoothing task but means the GNN component cannot be independently validated on held-out geographic areas within the current dataset.

REPLICABILITY

The methodology requires three inputs available globally: (1) road network with functional classification — Overture Maps updates monthly for every country; (2) GPS probe speed data — TomTom Move, HERE Traffic, or Google Roads API; (3) street imagery — Mapillary has global crowdsourced coverage. In data-scarce environments, Stage 1 alone produces actionable outputs, and 85th-percentile speeds can be estimated from road geometry features using regression models trained on the Thailand/Maharashtra data. The full pipeline, including all code and documentation, is available in the project repository.
```

---

**Findings Summary** *(what the model found + policy implications)*
```
WHAT THE MODEL FOUND

Thailand (11,134 segments analyzed):
- Grade E Critical: 1 segment — an urban primary road with a 90 km/h posted limit, 115.5 km/h 85th-percentile operating speed, and a Safe System threshold of 50 km/h. Estimated fatality reduction if corrected: 82%.
- Grade D Unsafe: 829 segments — predominantly urban primary and secondary roads where posted limits of 80–90 km/h exceed Safe System thresholds of 40–50 km/h by 30–50 km/h. High traffic volumes (many in the 60th–90th percentile nationally) mean millions of vehicle-kilometres of exposure annually on these unsafe corridors.
- Grade C Caution: 3,972 segments — roads with meaningful but less severe misalignment; these represent the pipeline for future Grade D classifications as traffic grows.
- The dominant pattern: Thailand's urban default speed limits are systematically above what Safe System principles recommend. This is not a problem of isolated bad roads — it is a policy-level misalignment that affects the majority of high-volume urban corridors.

Maharashtra (3,577 segments analyzed):
- Grade D Unsafe: 2 segments — urban trunk roads with 80 km/h limits where traffic operates at 98–104 km/h against a 60 km/h Safe System threshold. Estimated fatality reductions if corrected: 42–50%.
- Grade C Caution: 145 segments — rural and urban roads where operating speeds exceed Safe System thresholds by 20–40 km/h. Many of these are rural trunk roads that likely serve as key inter-city corridors.
- Critical context: Maharashtra's helmet-wearing rate for motorcycle passengers is 1.2%. At the speeds observed on Grade C/D segments, the probability of fatality in a crash involving an unhelmeted passenger approaches certainty. Speed management and helmet enforcement must be addressed together.

HOW THIS CAN BE USED BY TRANSPORT MINISTRIES

1. Prioritized intervention list: The model produces a ranked list of every road segment by Speed Safety Score, with associated road name, GPS coordinates, posted limit, recommended limit, and estimated fatality reduction. A transport ministry can take this list directly into a budget planning process and fund speed limit reviews for the top 50, 100, or 500 segments.

2. Evidence for speed limit reform: The data demonstrates that Thailand's urban speed limits are not an isolated compliance problem — they are a legislative and policy problem. A national review of the Highway Traffic Act's urban speed provisions, supported by this segment-level evidence, has a clear evidence base.

3. Enforcement targeting: Segments with high PercentOverLimit (many vehicles exceeding the posted limit) combined with high Speed Safety Scores identify locations where speed cameras or average-speed enforcement would have the greatest impact per unit of investment.

4. Infrastructure investment prioritization: VLM analysis identifies segments where pedestrian and cyclist infrastructure is absent on high-speed roads — exactly the locations where pedestrian bridges, protected crossings, or physical separation would save lives.

WHERE GOVERNMENTS SHOULD PRIORITIZE

Immediate (top 10 segments by impact index): The 10 segments with the highest combination of Speed Safety Score and traffic volume represent the greatest potential lives saved per intervention. These are all urban primary and secondary roads in Thailand. Reducing posted limits and deploying enforcement on these 10 segments alone is estimated to prevent more fatalities than any other equivalent investment in the dataset.

Systemic reform (Thailand urban primary roads): The 829 Grade D segments are not 829 separate problems — they are one problem: Thailand's urban primary road speed limit policy. A single policy decision to align urban primary road limits with Safe System recommendations would address the majority of the risk identified in this study.

Low-data environments: In countries where GPS probe data is unavailable, the Stage 1 component can be run with only road network data and estimated speeds (using road-type defaults or field surveys), still producing directionally correct priority rankings with less precision.
```

---

**Motivation**
```
I came to this challenge from an unexpected direction. I am a software engineer and AI developer, not a transport planner or road safety specialist. But the problem it poses — can AI tell us where speed limits need to change? — is exactly the kind of question I find most compelling: one where the technical challenge and the human stakes are both genuinely high.

Road safety is one of the few major global health crises where the solutions are well understood but implementation lags badly. The Safe System framework has existed for decades. The data to identify dangerous roads exists. The gap is in the tools to turn that data into actionable, prioritized, evidence-based decisions at the scale governments need to operate.

That is a software and AI problem as much as it is a policy problem. And it is a problem I knew I could make a meaningful contribution to.

I was also drawn by the specific challenge of working with multimodal data at real-world scale — combining GPS probe data, street-level imagery, and network topology into a single coherent score is not a trivial technical problem. The opportunity to apply a Vision-Language Model to analyse Mapillary imagery at the scale of an entire national road network, and to use Graph Attention Networks to capture spatial risk propagation, pushed me to build something genuinely novel rather than reaching for an off-the-shelf solution.

Finally, I believe this kind of work matters disproportionately in Asia and the Pacific, where road fatality rates are among the highest in the world, where motorcycle use is ubiquitous, and where the gap between current speed limits and Safe System standards is widest. If this methodology can help one government make one better decision about speed limits on one corridor, it will have been worth it. If it can be scaled across ADB member countries and embedded in ADB's data infrastructure — which this submission is designed to enable — it could contribute to saving tens of thousands of lives.
```

---

## DELIVERABLES
*(Fill these in once GitHub repo is published)*

**Analytical Model (GitHub link):**
```
https://github.com/[YOUR_USERNAME]/adb-safer-roads
```

**Speed Safety Score (GitHub link):**
```
https://github.com/[YOUR_USERNAME]/adb-safer-roads
```

**Geospatial Visualization (GitHub link):**
```
https://github.com/[YOUR_USERNAME]/adb-safer-roads
```

---

## OTHER

**How did you hear about this challenge?**
```
University / Academic Network
```

**Interest to work on a pilot with ADB:**
```
Yes, we would be available and interested
```
