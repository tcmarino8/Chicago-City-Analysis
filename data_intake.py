"""Data ingestion and preprocessing utilities for the Chicago AQI workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core import api_client, config, geo_utils


@dataclass
class PreparedData:
    time_series: pd.DataFrame
    sensor_catalog: pd.DataFrame


_PREPARED_CACHE: PreparedData | None = None


def load_raw_data(**kwargs) -> pd.DataFrame:
    """Wrapper around the Socrata loader with pagination."""
    return api_client.load_data(**kwargs)


def get_prepared_data(refresh: bool = False, **load_kwargs) -> PreparedData:
    global _PREPARED_CACHE
    if refresh or _PREPARED_CACHE is None:
        raw = load_raw_data(refresh=refresh, **load_kwargs)
        _PREPARED_CACHE = prepare_data(raw)
    return _PREPARED_CACHE


def prepare_data(
    df: pd.DataFrame,
    aqi_column: str = config.DEFAULT_AQI_COLUMN,
    time_column: str = config.DEFAULT_TIME_COLUMN,
    sensor_candidates: Iterable[str] = config.DEFAULT_SENSOR_CANDIDATES,
) -> PreparedData:
    df = geo_utils.ensure_lat_lon(df)
    df = df.copy()
    for col in sensor_candidates:
        if col in df.columns:
            df["sensor_name"] = df[col]
            break
    else:
        df["sensor_name"] = df.index.astype(str)

    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df[aqi_column] = pd.to_numeric(df[aqi_column], errors="coerce")

    df = df.dropna(subset=[time_column, aqi_column, "latitude", "longitude", "sensor_name"])
    df = df.rename(columns={time_column: "time", aqi_column: "aqi_value"})

    sensor_catalog = (
        df.sort_values("time")
        .groupby("sensor_name")
        .tail(1)
        .loc[:, ["sensor_name", "latitude", "longitude"]]
        .reset_index(drop=True)
    )

    return PreparedData(time_series=df.reset_index(drop=True), sensor_catalog=sensor_catalog)


def aggregate_time_series(df: pd.DataFrame, bin_code: str) -> pd.DataFrame:
    df = df.copy()
    df["time_bin"] = df["time"].dt.floor(bin_code)
    aggregated = (
        df.groupby(["sensor_name", "time_bin", "latitude", "longitude"])
        .agg(aqi_value=("aqi_value", "mean"), reading_count=("aqi_value", "size"))
        .reset_index()
        .rename(columns={"time_bin": "time"})
        .sort_values("time")
    )
    return aggregated


def compute_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("sensor_name")
        .aqi_value.agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    quantiles = df.groupby("sensor_name")["aqi_value"].quantile([0.1, 0.5, 0.9]).unstack()
    quantiles.columns = ["q10", "q50", "q90"]
    return grouped.merge(quantiles, on="sensor_name", how="left")


def compute_time_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("time")
        .aqi_value.agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )


def filter_by_time(df: pd.DataFrame, time_value: pd.Timestamp) -> pd.DataFrame:
    return df[df["time"] == time_value].reset_index(drop=True)


def compute_uptime(
    aggregated_df: pd.DataFrame,
    sensor_catalog: pd.DataFrame,
    time_value: pd.Timestamp,
    coverage_threshold: float = config.DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, float]]:
    slice_df = filter_by_time(aggregated_df, time_value)
    merged = sensor_catalog.merge(slice_df, on="sensor_name", how="left", suffixes=("", "_reading"))
    merged["is_observing"] = merged["aqi_value"].notna()
    stats = {
        "observing": int(merged["is_observing"].sum()),
        "not_observing": int((~merged["is_observing"]).sum()),
        "coverage_pct": float(merged["is_observing"].mean() * 100.0),
        "bin": str(time_value),
    }
    stats.update(geo_utils.spatial_stats(sensor_catalog))
    if stats["coverage_pct"] < coverage_threshold * 100:
        stats["insufficient"] = True
    return merged, stats


def build_distance_matrix(sensor_catalog: pd.DataFrame) -> np.ndarray:
    return geo_utils.pairwise_distance_matrix(sensor_catalog)


def build_adjacency_matrix(sensor_catalog: pd.DataFrame, threshold_km: float = config.ADJACENCY_DISTANCE_THRESHOLD_KM) -> np.ndarray:
    distances = build_distance_matrix(sensor_catalog)
    adjacency = (distances <= threshold_km).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency
