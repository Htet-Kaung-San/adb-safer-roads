"""
Segment Archetype Clustering — unsupervised discovery of risk pattern archetypes.

Clusters Grade C/D/E segments into five named archetypes using KMeans on a
feature set that captures *why* a segment is risky, not just *how much*.

Archetypes:
  "Urban Speedway"         — legislatively unsafe limits, urban context
  "High-Volume Corridor"   — borderline excess but extreme traffic exposure
  "Infrastructure Void"    — high speeds + absent pedestrian/cyclist protection
  "Speed Creep Zone"       — acceptable limit but high non-compliance rate
  "Rural Risk Corridor"    — rural high-speed segments with VRU presence

The cluster → archetype mapping is determined by matching cluster centroids to
archetype profile signatures, rather than by fixed cluster IDs. This makes the
assignment stable even if KMeans initialises differently across runs.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

from src.config import ARCHETYPE_PROFILES


# ── Feature importance per archetype ──────────────────────────────────────
# For each archetype, define a signature as a dict of (feature → expected normalised value).
# High value means the archetype has a strong signal on that feature.
# We match each cluster centroid to the closest signature.

ARCHETYPE_SIGNATURES: Dict[str, Dict[str, float]] = {
    "Urban Speedway": {
        "sub_posted_limit_excess": 0.90,   # Posted limit is itself very high
        "sub_speed_deviation":     0.70,
        "sub_traffic_exposure":    0.60,
        "sub_speeding_prevalence": 0.40,
        "lu_urban":                1.00,   # Urban context
        "sub_vru_vulnerability":   0.40,
    },
    "High-Volume Corridor": {
        "sub_traffic_exposure":    0.95,   # Extreme traffic is the defining feature
        "sub_speed_deviation":     0.45,
        "sub_posted_limit_excess": 0.40,
        "sub_speeding_prevalence": 0.50,
        "lu_urban":                0.60,
        "sub_vru_vulnerability":   0.35,
    },
    "Infrastructure Void": {
        "pedestrian_infra":        0.10,   # Very low pedestrian infra (VLM)
        "cyclist_infra":           0.10,
        "vru_exposure":            0.75,   # But VRU are present
        "roadside_activity":       0.70,
        "sub_speed_deviation":     0.60,
        "sub_vru_vulnerability":   0.60,
    },
    "Speed Creep Zone": {
        "sub_speeding_prevalence": 0.90,   # High non-compliance is defining
        "sub_posted_limit_excess": 0.20,   # Limit is not the problem
        "sub_speed_deviation":     0.55,
        "sub_traffic_exposure":    0.45,
        "signage_quality":         0.30,   # Often poor signage
        "sub_vru_vulnerability":   0.40,
    },
    "Rural Risk Corridor": {
        "lu_urban":                0.05,   # Rural context
        "sub_speed_deviation":     0.65,
        "sub_vru_vulnerability":   0.70,   # High vulnerability
        "sub_traffic_exposure":    0.30,   # Lower volume than urban
        "visibility_quality":      0.60,
        "sub_posted_limit_excess": 0.50,
    },
}

# Core features always used (VLM features added if available)
CORE_FEATURES = [
    "sub_speed_deviation",
    "sub_posted_limit_excess",
    "sub_speeding_prevalence",
    "sub_traffic_exposure",
    "sub_vru_vulnerability",
]

VLM_FEATURES_FOR_CLUSTERING = [
    "pedestrian_infra",
    "cyclist_infra",
    "roadside_activity",
    "signage_quality",
    "vru_exposure",
    "visibility_quality",
]

LAND_USE_FEATURES = ["lu_urban"]   # 1=urban, 0=rural


def _build_clustering_features(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, list]:
    """
    Build normalised feature matrix for clustering.
    Returns (X, feature_names).
    """
    feature_names = list(CORE_FEATURES)

    # Land use binary
    lu_urban = (gdf["LandUse"] == "URBAN").astype(float) if "LandUse" in gdf.columns \
               else pd.Series(0.5, index=gdf.index)

    parts = [gdf[CORE_FEATURES].fillna(0).values, lu_urban.values.reshape(-1, 1)]
    feature_names.append("lu_urban")

    # VLM features if available (fills gap with 0.5 = neutral)
    vlm_available = [c for c in VLM_FEATURES_FOR_CLUSTERING if c in gdf.columns]
    if vlm_available:
        vlm_data = gdf[vlm_available].fillna(0.5).values
        parts.append(vlm_data)
        feature_names.extend(vlm_available)

    X = np.hstack(parts).astype(np.float32)
    return X, feature_names


def _match_clusters_to_archetypes(
    centroids: np.ndarray,
    feature_names: list,
) -> Dict[int, str]:
    """
    Match each KMeans cluster centroid to the closest archetype signature.
    Uses cosine similarity on shared features.
    Returns {cluster_id → archetype_name}.
    """
    archetype_names = list(ARCHETYPE_SIGNATURES.keys())
    n_clusters = len(centroids)

    # Build signature matrix aligned to feature_names
    sig_matrix = np.zeros((len(archetype_names), len(feature_names)))
    for j, arch in enumerate(archetype_names):
        sig = ARCHETYPE_SIGNATURES[arch]
        for i, fname in enumerate(feature_names):
            sig_matrix[j, i] = sig.get(fname, 0.5)

    # Cosine similarity between each centroid and each signature
    def _cosine(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # Hungarian-style greedy matching: assign each cluster to its best unmatched archetype
    similarity = np.array([
        [_cosine(centroids[i], sig_matrix[j]) for j in range(len(archetype_names))]
        for i in range(n_clusters)
    ])

    used_archetypes = set()
    cluster_to_archetype = {}

    # Sort assignments by similarity (highest first)
    assignments = sorted(
        [(i, j, similarity[i, j]) for i in range(n_clusters) for j in range(len(archetype_names))],
        key=lambda x: -x[2],
    )
    for cluster_id, arch_idx, sim in assignments:
        if cluster_id in cluster_to_archetype:
            continue
        arch_name = archetype_names[arch_idx]
        if arch_name in used_archetypes and n_clusters <= len(archetype_names):
            continue
        cluster_to_archetype[cluster_id] = arch_name
        used_archetypes.add(arch_name)

    # Fill any unassigned clusters with the default
    for i in range(n_clusters):
        if i not in cluster_to_archetype:
            cluster_to_archetype[i] = "Urban Speedway"

    return cluster_to_archetype


def assign_archetypes(
    gdf: gpd.GeoDataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
    grade_filter: Tuple[str, ...] = ("C", "D", "E"),
) -> gpd.GeoDataFrame:
    """
    Cluster Grade C/D/E segments into archetypes and annotate the full GeoDataFrame.

    Segments not in grade_filter (i.e., safe A/B segments) receive:
        archetype_name = "Not Applicable", archetype_description = ""

    Args:
        gdf: GeoDataFrame with speed_safety_score (or final_score) + sub-scores
        n_clusters: Number of KMeans clusters (should match number of archetypes = 5)
        random_state: KMeans seed for reproducibility
        grade_filter: Only cluster segments with these grades

    Returns:
        gdf with new columns: archetype_name, archetype_description,
                              archetype_intervention, archetype_color, archetype_icon
    """
    gdf = gdf.copy()

    grade_col = "final_grade" if "final_grade" in gdf.columns else "score_grade"
    risk_mask = gdf[grade_col].isin(grade_filter)
    risk_gdf = gdf[risk_mask].copy()

    if len(risk_gdf) < n_clusters:
        print(f"  Warning: only {len(risk_gdf)} risk segments, fewer than n_clusters={n_clusters}. "
              f"Using {max(1, len(risk_gdf))} clusters.")
        n_clusters = max(1, len(risk_gdf))

    X, feature_names = _build_clustering_features(risk_gdf)

    # Standardise features so KMeans is not dominated by scale differences
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)

    # Transform centroids back to original scale for archetype matching
    centroids_original = scaler.inverse_transform(km.cluster_centers_)
    cluster_to_archetype = _match_clusters_to_archetypes(centroids_original, feature_names)

    print(f"  Cluster → Archetype mapping:")
    for cid, aname in sorted(cluster_to_archetype.items()):
        count = (labels == cid).sum()
        print(f"    Cluster {cid} → '{aname}' ({count} segments)")

    # Map cluster labels to archetype names
    archetype_labels = [cluster_to_archetype[lbl] for lbl in labels]
    risk_gdf["archetype_name"] = archetype_labels

    # Propagate full profile info
    for col, getter in [
        ("archetype_description",   lambda n: ARCHETYPE_PROFILES.get(n, {}).get("description", "")),
        ("archetype_intervention",  lambda n: ARCHETYPE_PROFILES.get(n, {}).get("primary_intervention", "")),
        ("archetype_secondary",     lambda n: ARCHETYPE_PROFILES.get(n, {}).get("secondary_intervention", "")),
        ("archetype_color",         lambda n: ARCHETYPE_PROFILES.get(n, {}).get("color", "#666")),
        ("archetype_icon",          lambda n: ARCHETYPE_PROFILES.get(n, {}).get("icon", "")),
    ]:
        risk_gdf[col] = risk_gdf["archetype_name"].map(getter)

    # Merge back into the full GeoDataFrame
    for col in ["archetype_name", "archetype_description", "archetype_intervention",
                "archetype_secondary", "archetype_color", "archetype_icon"]:
        gdf[col] = "Not Applicable"

    gdf.loc[risk_mask, "archetype_name"]         = risk_gdf["archetype_name"].values
    gdf.loc[risk_mask, "archetype_description"]  = risk_gdf["archetype_description"].values
    gdf.loc[risk_mask, "archetype_intervention"] = risk_gdf["archetype_intervention"].values
    gdf.loc[risk_mask, "archetype_secondary"]    = risk_gdf["archetype_secondary"].values
    gdf.loc[risk_mask, "archetype_color"]        = risk_gdf["archetype_color"].values
    gdf.loc[risk_mask, "archetype_icon"]         = risk_gdf["archetype_icon"].values

    # Summary
    for arch in ARCHETYPE_PROFILES:
        count = (gdf["archetype_name"] == arch).sum()
        if count > 0:
            print(f"    {arch}: {count} segments")

    return gdf


def get_archetype_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Summary table: archetype × (count, avg_score, avg_economic_benefit,
                                 avg_nilsson, primary_intervention).
    """
    if "archetype_name" not in gdf.columns:
        return pd.DataFrame()

    rows = []
    for arch in ARCHETYPE_PROFILES:
        mask = gdf["archetype_name"] == arch
        sub = gdf[mask]
        if sub.empty:
            continue
        score_col = "final_score" if "final_score" in sub.columns else "speed_safety_score"
        rows.append({
            "archetype":              arch,
            "count":                  len(sub),
            "avg_final_score":        sub[score_col].mean().round(1),
            "avg_nilsson_pct":        (sub.get("nilsson_reduction_factor", 0) * 100).mean().round(1),
            "avg_economic_benefit_m": sub.get("economic_benefit_m", pd.Series(0)).mean().round(2),
            "total_economic_benefit_m": sub.get("economic_benefit_m", pd.Series(0)).sum().round(2),
            "primary_intervention":   ARCHETYPE_PROFILES[arch]["primary_intervention"],
            "icon":                   ARCHETYPE_PROFILES[arch]["icon"],
        })

    return pd.DataFrame(rows).sort_values("total_economic_benefit_m", ascending=False)
