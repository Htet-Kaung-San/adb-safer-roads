"""
Mapillary API client — fetch street-level images near road segment midpoints.

Usage:
    client = MapillaryClient(token=os.environ["MAPILLARY_TOKEN"])
    images = client.fetch_images_for_segment(lat=14.94, lon=103.48, radius_m=50)
    client.download_images(images, out_dir=Path("data/processed/mapillary_cache/segment_123"))
"""
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.config import MAPILLARY_RADIUS_M, MAPILLARY_MAX_IMAGES, MAPILLARY_CACHE_DIR

logger = logging.getLogger(__name__)

MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"


class MapillaryClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("MAPILLARY_TOKEN")
        if not self.token:
            raise ValueError(
                "Provide a Mapillary access token via MAPILLARY_TOKEN env var "
                "or the token= argument. Get one at mapillary.com → Dashboard → Developers."
            )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"OAuth {self.token}"

    def _get(self, url: str, params: dict, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=15)
                # 500 from Mapillary = no coverage at this location; treat as empty
                if r.status_code == 500:
                    return {}
                r.raise_for_status()
                return r.json()
            except requests.HTTPError:
                return {}
            except requests.RequestException:
                if attempt == retries - 1:
                    return {}
                time.sleep(2 ** attempt)
        return {}

    def fetch_images_for_segment(
        self,
        lat: float,
        lon: float,
        radius_m: int = MAPILLARY_RADIUS_M,
        max_images: int = MAPILLARY_MAX_IMAGES,
    ) -> list[dict]:
        """
        Return up to max_images Mapillary image records near (lat, lon).

        Each record contains: id, thumb_1024_url, compass_angle, captured_at.
        """
        params = {
            "fields": "id,thumb_1024_url,compass_angle,captured_at,geometry",
            "bbox": self._bbox(lat, lon, radius_m),
            "limit": max_images,
        }
        data = self._get(f"{MAPILLARY_GRAPH_URL}/images", params)
        return data.get("data", [])

    def download_images(
        self,
        images: list[dict],
        out_dir: Path,
        num_threads: int = 4,
    ) -> list[Path]:
        """Download images to out_dir; skip already-cached files. Returns file paths."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        def _download_one(img: dict) -> Optional[Path]:
            url = img.get("thumb_1024_url")
            if not url:
                return None
            fname = out_dir / f"{img['id']}.jpg"
            if fname.exists():
                return fname
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                fname.write_bytes(r.content)
                return fname
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {ex.submit(_download_one, img): img for img in images}
            for fut in as_completed(futures):
                p = fut.result()
                if p:
                    paths.append(p)

        return paths

    def fetch_and_cache_segment(
        self,
        segment_id: str,
        lat: float,
        lon: float,
        radius_m: int = MAPILLARY_RADIUS_M,
        max_images: int = MAPILLARY_MAX_IMAGES,
    ) -> list[Path]:
        """
        Convenience: fetch + download images for one segment, using a
        cache directory keyed by segment_id. Returns local file paths.
        Retries with progressively larger radius if no images found.
        """
        cache_dir = MAPILLARY_CACHE_DIR / str(segment_id)
        existing = list(cache_dir.glob("*.jpg")) if cache_dir.exists() else []
        if existing:
            return existing

        # Try increasing radii: 50m → 150m → 500m
        for r in [radius_m, radius_m * 3, radius_m * 10]:
            images = self.fetch_images_for_segment(lat, lon, r, max_images)
            if images:
                return self.download_images(images, cache_dir)

        logger.debug(f"No Mapillary imagery for segment {segment_id} ({lat:.5f}, {lon:.5f})")
        return []

    @staticmethod
    def _bbox(lat: float, lon: float, radius_m: int) -> str:
        """Approximate bounding box string for the Mapillary API."""
        deg = radius_m / 111_320.0
        return f"{lon-deg},{lat-deg},{lon+deg},{lat+deg}"


def batch_fetch_images(
    gdf,
    token: str,
    num_threads: int = 16,
    max_images_per_segment: int = MAPILLARY_MAX_IMAGES,
) -> dict[str, list[Path]]:
    """
    Fetch and cache Mapillary images for all segments in gdf in parallel.

    Returns: {segment_id: [local_image_paths]}
    """
    client = MapillaryClient(token=token)
    results = {}

    def _fetch(row):
        seg_id = str(row.get("OBJECTID", row.name))
        paths = client.fetch_and_cache_segment(
            segment_id=seg_id,
            lat=row["mid_lat"],
            lon=row["mid_lon"],
            max_images=max_images_per_segment,
        )
        return seg_id, paths

    rows = [row for _, row in gdf.iterrows()]
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {ex.submit(_fetch, row): row for row in rows}
        done = 0
        for fut in as_completed(futures):
            seg_id, paths = fut.result()
            results[seg_id] = paths
            done += 1
            if done % 500 == 0:
                logger.info(f"Mapillary fetch: {done}/{len(rows)} segments done")

    cached = sum(1 for p in results.values() if p)
    logger.info(f"Mapillary: {cached}/{len(results)} segments have imagery")
    return results
