"""Spatial interpolation utilities (IDW, Kriging, etc.)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
from scipy.spatial import distance

from core import config, geo_utils


@dataclass
class InterpolationResult:
    grid_lon: np.ndarray
    grid_lat: np.ndarray
    grid_values: np.ndarray
    method_used: str
    source_points: pd.DataFrame


def _build_grid(df_slice: pd.DataFrame, grid_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    bbox = geo_utils.BoundingBox.from_points(df_slice["latitude"], df_slice["longitude"], padding=0.1)
    lon = np.linspace(bbox.lon_min, bbox.lon_max, grid_resolution)
    lat = np.linspace(bbox.lat_min, bbox.lat_max, grid_resolution)
    return np.meshgrid(lon, lat)


def _build_grid_from_bbox(bbox: geo_utils.BoundingBox, grid_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(bbox.lon_min, bbox.lon_max, grid_resolution)
    lat = np.linspace(bbox.lat_min, bbox.lat_max, grid_resolution)
    return np.meshgrid(lon, lat)


def _idw(points: np.ndarray, values: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray, power: float = 2.0) -> np.ndarray:
    grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    distances = distance.cdist(grid_points, points)
    distances[distances == 0] = 1e-10
    weights = 1.0 / np.power(distances, power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    interpolated = (weights / weights_sum) @ values
    return interpolated.reshape(grid_lon.shape)


def _kriging(points: np.ndarray, values: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    ok = OrdinaryKriging(
        points[:, 0],
        points[:, 1],
        values,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        nlags=6,
    )
    z, _ = ok.execute("grid", grid_lon[0, :], grid_lat[:, 0])
    return z


def _linear(points: np.ndarray, values: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    grid_values = griddata(points, values, (grid_lon, grid_lat), method="linear")
    if np.isnan(grid_values).any():
        nearest = griddata(points, values, (grid_lon, grid_lat), method="nearest")
        grid_values = np.where(np.isnan(grid_values), nearest, grid_values)
    return grid_values


def interpolate_time_slice(
    df_slice: pd.DataFrame,
    method: Literal["idw", "kriging", "linear"] = "idw",
    grid_resolution: int = config.DEFAULT_GRID_RESOLUTION,
    bbox: geo_utils.BoundingBox | None = None,
) -> InterpolationResult | None:
    if df_slice.shape[0] < 3:
        return None
    if bbox is None:
        grid_lon, grid_lat = _build_grid(df_slice, grid_resolution)
    else:
        grid_lon, grid_lat = _build_grid_from_bbox(bbox, grid_resolution)
    points = df_slice[["longitude", "latitude"]].to_numpy(dtype=float)
    values = df_slice["aqi_value"].to_numpy(dtype=float)
    method_used = method
    try:
        if method == "idw":
            grid_values = _idw(points, values, grid_lon, grid_lat)
        elif method == "kriging":
            grid_values = _kriging(points, values, grid_lon, grid_lat)
        elif method == "linear":
            grid_values = _linear(points, values, grid_lon, grid_lat)
        else:
            raise ValueError(f"Unsupported method: {method}")
    except Exception:
        method_used = "idw"
        grid_values = _idw(points, values, grid_lon, grid_lat)
    return InterpolationResult(grid_lon, grid_lat, grid_values, method_used, df_slice)


class STGNNModel:
    """Placeholder for future spatiotemporal GNN work."""

    def __init__(self, *_, **__):
        raise NotImplementedError("STGNN model integration will be implemented after network analysis is complete.")
