"""
Central configuration: paths, Safe System thresholds, scoring weights, economic constants.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
SCORES_DIR = OUTPUTS / "scores"
MAPS_DIR = OUTPUTS / "maps"

# Data files live in the project root (alongside the code)
MAHARASHTRA_GEOJSON = ROOT / "ADB_Innovation_Maharashtra.geojson"
THAILAND_GEOJSON = ROOT / "ADB_Innovation_Thailand.geojson"
HELMET_SPI_XLSX = ROOT / "Archive" / "Road_Safety_Performance_Indicators_(Helmet_Wearing_results)_(adb_dashboard_data_v02).xlsx"

# ── Safe System Speed Thresholds (km/h) ───────────────────────────────────
# WHO Safe System principles and UNECE road safety targets.
# Represents the maximum speed at which the road type/context is safe for
# all road users including pedestrians and cyclists.
SAFE_SYSTEM_THRESHOLDS = {
    # (road_class, land_use) -> safe speed in km/h
    ("motorway",  "RURAL"):  110,
    ("motorway",  "URBAN"):   80,
    ("trunk",     "RURAL"):   80,
    ("trunk",     "URBAN"):   60,
    ("primary",   "RURAL"):   80,
    ("primary",   "URBAN"):   50,
    ("secondary", "RURAL"):   60,
    ("secondary", "URBAN"):   40,
    # Fallbacks when one dimension is unknown
    ("motorway",  None):     100,
    ("trunk",     None):      70,
    ("primary",   None):      60,
    ("secondary", None):      50,
    (None,        "RURAL"):   80,
    (None,        "URBAN"):   50,
    (None,        None):      60,
}

# ── Helmet SPI — VRU Vulnerability Multiplier ─────────────────────────────
# All-rider helmet wearing rates from XLSX. Lower rate = higher vulnerability.
HELMET_SPI = {
    "Thailand":    0.778,
    "Maharashtra": 0.209,
    "Mumbai":      0.555,
    "Pune":        0.213,
}

# Baseline (best-practice) helmet rate to normalise against
HELMET_BASELINE = 0.90

# ── Speed Safety Score Weights (must sum to 100) ──────────────────────────
SCORE_WEIGHTS = {
    "speed_deviation":      40,   # How far 85th-percentile exceeds Safe System threshold
    "posted_limit_excess":  20,   # How far posted limit exceeds Safe System threshold
    "speeding_prevalence":  20,   # Percent of vehicles over posted limit
    "traffic_exposure":     10,   # Traffic volume (more traffic = more lives at risk)
    "vru_vulnerability":    10,   # Inverse of helmet SPI (unprotected road users)
}

# ── Score Classification ───────────────────────────────────────────────────
SCORE_BANDS = [
    (0,  20,  "A", "Safe",     "#2ecc71"),
    (20, 40,  "B", "Adequate", "#f1c40f"),
    (40, 60,  "C", "Caution",  "#e67e22"),
    (60, 80,  "D", "Unsafe",   "#e74c3c"),
    (80, 100, "E", "Critical", "#8e44ad"),
]

# ── Nilsson Power Model Exponents ─────────────────────────────────────────
NILSSON_EXPONENTS = {
    "fatality":       4.0,
    "serious_injury": 3.0,
    "all_injury":     2.0,
}

# ── Economic Impact Parameters ────────────────────────────────────────────
# Value of Statistical Life (USD) — used to monetise fatality reduction estimates.
# Sources: World Bank upper-middle income country estimates; India MoRTH IRC:SP:88.
VOSL_USD = {
    "thailand":    1_260_000,   # World Bank VSL for upper-middle income (Thailand 2023)
    "maharashtra":   420_000,   # India MoRTH IRC:SP:88 adjusted to USD at 2024 rates
    "default":       800_000,
}

# Annual fatal crash rate per 100 million vehicle-kilometres.
# Sources: IRTAD 2022 (Thailand), NCRB Road Accidents in India 2022 (Maharashtra).
CRASH_RATE_PER_100M_VMT = {
    "thailand":    8.4,
    "maharashtra": 11.2,
    "default":      9.5,
}

# Traffic volume proxy: map RankedPercentile [0,1] → estimated vehicles/day.
# Log-linear: 0th percentile ≈ 200 veh/day, 100th percentile ≈ 60,000 veh/day.
# These are order-of-magnitude estimates for relative economic ranking only.
TRAFFIC_VOLUME_MIN_VPD = 200
TRAFFIC_VOLUME_MAX_VPD = 60_000

# ── Segment Archetype Profiles ────────────────────────────────────────────
# KMeans cluster → human-readable archetype with intervention guidance.
# Cluster IDs are assigned by matching centroids to these profiles in clustering.py.
ARCHETYPE_PROFILES = {
    "Urban Speedway": {
        "description": "High posted limits on urban primary/secondary roads. Systemic policy misalignment — the limit itself is the problem.",
        "primary_intervention": "National speed limit policy reform + average speed enforcement",
        "secondary_intervention": "Urban traffic management centres with variable speed signs",
        "color": "#e74c3c",
        "icon": "🏙️",
    },
    "High-Volume Corridor": {
        "description": "Borderline speed excess, but extreme traffic exposure makes every crash statistically likely. Risk scales with volume.",
        "primary_intervention": "Peak-hour variable speed limits + congestion-responsive enforcement",
        "secondary_intervention": "Grade-separated pedestrian crossings at highest-volume points",
        "color": "#e67e22",
        "icon": "🚗",
    },
    "Infrastructure Void": {
        "description": "Operating speeds that would be marginal on a protected road become lethal because pedestrian and cyclist infrastructure is absent.",
        "primary_intervention": "Physical separation: pedestrian bridges, crash barriers, protected crossings",
        "secondary_intervention": "Interim: kerb extensions, raised crossings, road diet",
        "color": "#9b59b6",
        "icon": "🚶",
    },
    "Speed Creep Zone": {
        "description": "Posted limit is reasonable relative to Safe System standards, but a large share of drivers routinely exceed it by 20-35 km/h. Enforcement gap.",
        "primary_intervention": "Speed camera deployment (fixed + average) + signage visibility audit",
        "secondary_intervention": "Road surface treatments (rumble strips, optical narrowing) to reduce speeds",
        "color": "#3498db",
        "icon": "⚡",
    },
    "Rural Risk Corridor": {
        "description": "Rural alignment encourages high speeds; vulnerable road users (motorcyclists, pedestrians) use the road in a context with no separation.",
        "primary_intervention": "Roadside safety treatment: barriers, clear zones, hazard removal",
        "secondary_intervention": "Collision warning systems + periodic safety rest areas",
        "color": "#27ae60",
        "icon": "🌾",
    },
}

# ── Mapillary ─────────────────────────────────────────────────────────────
MAPILLARY_RADIUS_M = 50          # search radius around segment midpoint
MAPILLARY_MAX_IMAGES = 3         # images per segment (averaged for robustness)
MAPILLARY_CACHE_DIR = DATA_PROCESSED / "mapillary_cache"

# ── VLM ───────────────────────────────────────────────────────────────────
VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"   # swap to 72B on cluster
VLM_BATCH_SIZE = 16                         # per GPU; scale with VRAM
NUM_GPUS = 8                                # RTX A5000 × 8

# ── GNN ───────────────────────────────────────────────────────────────────
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_HEADS = 4                # GAT attention heads
GNN_DROPOUT = 0.2
GNN_EPOCHS = 100
GNN_LR = 1e-3
SPATIAL_SNAP_DISTANCE_M = 20  # max distance to link segment endpoints
GNN_MC_SAMPLES = 50           # Monte Carlo dropout passes for uncertainty estimation
