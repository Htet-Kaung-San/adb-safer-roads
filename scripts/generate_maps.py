"""
Script: Generate all interactive maps.

Usage:
    python scripts/generate_maps.py

Outputs:
    outputs/maps/maharashtra_map.html
    outputs/maps/thailand_map.html
    outputs/maps/combined_map.html
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.dashboard import build_region_maps
from src.config import MAPS_DIR

MAPS_DIR.mkdir(parents=True, exist_ok=True)
build_region_maps()
print("\nAll maps generated.")
