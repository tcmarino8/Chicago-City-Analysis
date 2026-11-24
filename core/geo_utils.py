# python"""Geospatial helpers shared across the project."""
from __future__ import annotations

from dataclasses import dataclass
from json import loads as json_loads
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class BoundingBox:
    """Simple bounding box container."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_points(cls, lats: Iterable[float], lons: Iterable[float], padding: float = 0.1) -> "BoundingBox":
        lat_arr = np.asarray(list(lats), dtype=float)
        lon_arr = np.asarray(list(lons), dtype=float)
        if lat_arr.size == 0 or lon_arr.size == 0:
            raise ValueError("Cannot build bounding box from empty coordinate arrays")
        lat_span = lat_arr.max() - lat_arr.min()
        lon_span = lon_arr.max() - lon_arr.min()
        return cls(
            lat_min=float(lat_arr.min() - lat_span * padding),
            lat_max=float(lat_arr.max() + lat_span * padding),
            lon_min=float(lon_arr.min() - lon_span * padding),
            lon_max=float(lon_arr.max() + lon_span * padding),
        )


def extract_coordinates(location_value) -> Tuple[float | None, float | None]:
    """Return (lat, lon) pairs from the heterogeneous Socrata location field."""
    if pd.isna(location_value):
        return None, None
    if isinstance(location_value, dict):
        if {"latitude", "longitude"} <= location_value.keys():
            return float(location_value["latitude"]), float(location_value["longitude"])
        if "coordinates" in location_value:
            lon, lat = location_value["coordinates"][:2]
            return float(lat), float(lon)
    if isinstance(location_value, str):
        try:
            parsed = json_loads(location_value)
            return extract_coordinates(parsed)
        except Exception:
            return None, None
    return None, None


def ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee dataframe has float latitude/longitude columns."""
    df = df.copy()
    lat_col = config.DEFAULT_LAT_COLUMN
    lon_col = config.DEFAULT_LON_COLUMN
    if lat_col in df.columns and lon_col in df.columns:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        return df
    if "location" in df.columns:
        coords = df["location"].apply(extract_coordinates)
        df[lat_col] = coords.apply(lambda x: x[0])
        df[lon_col] = coords.apply(lambda x: x[1])
    else:
        df[lat_col] = np.nan
        df[lon_col] = np.nan
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    return df


def haversine_matrix(df_points: pd.DataFrame, lat_col: str = config.DEFAULT_LAT_COLUMN, lon_col: str = config.DEFAULT_LON_COLUMN) -> np.ndarray:
    """Return the condensed pairwise haversine distances (km)."""
    df_valid = df_points[[lat_col, lon_col]].dropna()
    coords = np.radians(df_valid[[lat_col, lon_col]].to_numpy())
    if coords.shape[0] < 2:
        return np.array([])
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    distances = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return distances[np.triu_indices(distances.shape[0], k=1)]


def spatial_stats(df_points: pd.DataFrame) -> dict[str, float]:
    """Compute range/mean/min spatial separations for reporting."""
    condensed = haversine_matrix(df_points)
    if condensed.size == 0:
        return {"range_km": 0.0, "mean_km": 0.0, "min_km": 0.0}
    return {
        "range_km": float(condensed.max()),
        "mean_km": float(condensed.mean()),
        "min_km": float(condensed.min()),
    }


def pairwise_distance_matrix(df_points: pd.DataFrame, lat_col: str = config.DEFAULT_LAT_COLUMN, lon_col: str = config.DEFAULT_LON_COLUMN) -> np.ndarray:
    """Return the full pairwise haversine distance matrix (km)."""
    coords = np.radians(df_points[[lat_col, lon_col]].to_numpy())
    if coords.shape[0] == 0:
        return np.empty((0, 0))
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
