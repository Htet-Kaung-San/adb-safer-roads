"""
Central configuration: paths, Safe System thresholds, scoring weights.
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
# Based on WHO Safe System principles and UNECE road safety targets.
# Thresholds represent the maximum speed at which the given road type/context
# is considered safe for all road users including pedestrians and cyclists.
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

# Baseline (best-practice) helmet rate to normalize against
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
