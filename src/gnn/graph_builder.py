"""
Build a road network graph from GeoDataFrame for GNN-based spatial scoring.

Nodes = road segments (LineStrings).
Edges = spatial connectivity (endpoints within SPATIAL_SNAP_DISTANCE_M metres).

Also encodes feature matrix X and score vector y for training/inference.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

from src.config import SPATIAL_SNAP_DISTANCE_M, SAFE_SYSTEM_THRESHOLDS

# Features used as GNN node inputs (all normalised to ~[0,1])
TABULAR_FEATURES = [
    "sub_speed_deviation",
    "sub_posted_limit_excess",
    "sub_speeding_prevalence",
    "sub_traffic_exposure",
    "sub_vru_vulnerability",
    "speed_safety_score",        # Stage 1 score as input feature
    "F85thPercentileSpeed",
    "MedianSpeed",
    "SpeedLimit",
    "length_km",
    "RankedPercentile",
]

VLM_FEATURES = [
    "pedestrian_infra",
    "cyclist_infra",
    "roadside_activity",
    "road_condition",
    "signage_quality",
    "vru_exposure",
    "visibility_quality",
]


def _road_class_ohe(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    classes = ["motorway", "trunk", "primary", "secondary"]
    ohe = pd.get_dummies(gdf["RoadClass"], prefix="rc")
    for c in classes:
        col = f"rc_{c}"
        if col not in ohe.columns:
            ohe[col] = 0
    return ohe[[f"rc_{c}" for c in classes]].astype(float)


def _land_use_ohe(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    ohe = pd.DataFrame({
        "lu_urban": (gdf["LandUse"] == "URBAN").astype(float),
        "lu_rural": (gdf["LandUse"] == "RURAL").astype(float),
    }, index=gdf.index)
    return ohe


def build_feature_matrix(gdf: gpd.GeoDataFrame, has_vlm: bool = False) -> np.ndarray:
    """
    Assemble the node feature matrix X of shape (N, F).

    Includes: tabular Stage-1 sub-scores + road class OHE + land use OHE
    + VLM features (if available).
    """
    parts = []

    # Tabular
    tab_cols = [c for c in TABULAR_FEATURES if c in gdf.columns]
    tab = gdf[tab_cols].fillna(0).copy()
    # Normalise speed columns to [0,1] by dividing by 130 km/h
    for col in ["F85thPercentileSpeed", "MedianSpeed", "SpeedLimit"]:
        if col in tab.columns:
            tab[col] = tab[col] / 130.0
    if "length_km" in tab.columns:
        tab["length_km"] = tab["length_km"].clip(0, 50) / 50.0
    if "RankedPercentile" in tab.columns:
        tab["RankedPercentile"] = tab["RankedPercentile"] / 100.0
    parts.append(tab.values)

    # Road class OHE
    parts.append(_road_class_ohe(gdf).values)

    # Land use OHE
    parts.append(_land_use_ohe(gdf).values)

    # VLM features (already 0-1)
    if has_vlm:
        vlm_cols = [c for c in VLM_FEATURES if c in gdf.columns]
        parts.append(gdf[vlm_cols].fillna(0.5).values)

    return np.hstack(parts).astype(np.float32)


def build_adjacency(gdf: gpd.GeoDataFrame, snap_distance_m: float = SPATIAL_SNAP_DISTANCE_M):
    """
    Build edge list (src, dst) by snapping segment endpoints.

    Strategy: extract all segment start/end points, build a KD-tree,
    find points within snap_distance_m, link their parent segments.
    Returns edge_index of shape (2, E) as numpy arrays.
    """
    # Project to UTM for accurate distance in metres
    # Use a single approximate UTM (works for Thailand/Maharashtra separately;
    # for combined dataset use a global equal-area projection)
    gdf_proj = gdf.to_crs("EPSG:32647")   # rough approximation for both regions

    # Collect all endpoints
    starts = np.array([(geom.coords[0][0], geom.coords[0][1]) for geom in gdf_proj.geometry])
    ends   = np.array([(geom.coords[-1][0], geom.coords[-1][1]) for geom in gdf_proj.geometry])

    all_points = np.vstack([starts, ends])   # shape (2N, 2)
    seg_ids    = np.tile(np.arange(len(gdf)), 2)  # which segment each point belongs to

    tree = cKDTree(all_points)
    pairs = tree.query_pairs(r=snap_distance_m, output_type="ndarray")  # shape (M, 2)

    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64)

    src_segs = seg_ids[pairs[:, 0]]
    dst_segs = seg_ids[pairs[:, 1]]

    # Remove self-loops
    mask = src_segs != dst_segs
    src_segs = src_segs[mask]
    dst_segs = dst_segs[mask]

    # Make bidirectional and deduplicate
    edges = np.stack([
        np.concatenate([src_segs, dst_segs]),
        np.concatenate([dst_segs, src_segs]),
    ])
    edges = np.unique(edges, axis=1)

    print(f"Graph: {len(gdf)} nodes, {edges.shape[1]} edges "
          f"(snap={snap_distance_m}m, avg degree={edges.shape[1]/len(gdf):.1f})")
    return edges


def build_graph(
    gdf: gpd.GeoDataFrame,
    has_vlm: bool = False,
    snap_distance_m: float = SPATIAL_SNAP_DISTANCE_M,
):
    """
    Build full graph data structure for PyG or DGL.

    Returns:
        x: np.ndarray (N, F)  node features
        edge_index: np.ndarray (2, E)
        y: np.ndarray (N,)    Stage-1 speed_safety_score (target for GNN refinement)
        node_index: pd.Index  maps row → original GDF index
    """
    x = build_feature_matrix(gdf, has_vlm=has_vlm)
    edge_index = build_adjacency(gdf, snap_distance_m=snap_distance_m)
    y = gdf["speed_safety_score"].fillna(50).values.astype(np.float32)
    return x, edge_index, y, gdf.index


def to_pyg_data(x, edge_index, y):
    """Convert numpy arrays to a PyTorch Geometric Data object."""
    import torch
    from torch_geometric.data import Data
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    )
