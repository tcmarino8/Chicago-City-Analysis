"""Chicago Open Data and Census overlay loaders used by the workspace map."""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from . import api_client

try:  # Optional dependency; available in most geo stacks.
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency
    gpd = None


CHICAGO_BOUNDS = {
    "lat_min": 41.60,
    "lat_max": 42.05,
    "lon_min": -87.95,
    "lon_max": -87.50,
}

ENERGY_DATASET_ID = "g5i5-yz37"
SCHOOLS_DATASET_ID = "3dhs-m3w4"
DEFAULT_CENSUS_YEAR = 2022


def _request_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "EODataChicagoAQI/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _in_chicago_bounds(lat: float, lon: float) -> bool:
    return (
        CHICAGO_BOUNDS["lat_min"] <= lat <= CHICAGO_BOUNDS["lat_max"]
        and CHICAGO_BOUNDS["lon_min"] <= lon <= CHICAGO_BOUNDS["lon_max"]
    )


@lru_cache(maxsize=8)
def _energy_rows(limit: int) -> pd.DataFrame:
    rows = api_client.fetch_rows(dataset_id=ENERGY_DATASET_ID, limit=limit)
    return pd.DataFrame.from_records(rows)


@lru_cache(maxsize=8)
def _schools_rows(limit: int) -> pd.DataFrame:
    rows = api_client.fetch_rows(dataset_id=SCHOOLS_DATASET_ID, limit=limit)
    return pd.DataFrame.from_records(rows)


def fetch_energy_overlay(metric: str = "energy_star_score", limit: int = 4000) -> dict[str, Any]:
    try:
        df = _energy_rows(limit=limit).copy()
        if df.empty:
            return {
                "available": False,
                "points": [],
                "summary": ["Energy overlay unavailable: dataset returned no rows."],
                "loaded": True,
            }

        if "latitude" not in df.columns or "longitude" not in df.columns:
            return {
                "available": False,
                "points": [],
                "summary": ["Energy overlay unavailable: latitude/longitude columns were not found."],
                "loaded": True,
            }

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        if metric not in df.columns:
            metric = "energy_star_score" if "energy_star_score" in df.columns else metric

        df["metric_value"] = pd.to_numeric(df.get(metric), errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "metric_value"])
        df = df[df.apply(lambda r: _in_chicago_bounds(float(r["latitude"]), float(r["longitude"])), axis=1)]

        points = []
        for row in df.itertuples(index=False):
            points.append(
                {
                    "name": getattr(row, "property_name", None) or getattr(row, "building_name", None) or "Building",
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                    "metric": metric,
                    "value": float(row.metric_value),
                    "address": getattr(row, "address", None) or getattr(row, "location_address", None),
                }
            )

        summary = [
            f"Dataset: Chicago Energy Benchmarking - Covered Buildings ({ENERGY_DATASET_ID})",
            f"Metric: {metric}",
            f"Points loaded: {len(points)}",
        ]
        if points:
            values = [p["value"] for p in points]
            summary.append(f"Mean value: {sum(values) / len(values):.2f}")

        return {
            "available": bool(points),
            "points": points,
            "summary": summary,
            "loaded": True,
            "metric": metric,
        }
    except Exception as exc:  # pragma: no cover - network safety
        return {
            "available": False,
            "points": [],
            "summary": [f"Energy overlay unavailable: {exc}"],
            "loaded": True,
            "metric": metric,
        }


def fetch_schools_overlay(
    metric: str = "graduation_rate_mean",
    school_type: str = "all",
    school_year: str = "all",
    limit: int = 5000,
) -> dict[str, Any]:
    try:
        df = _schools_rows(limit=limit).copy()
        if df.empty:
            return {
                "available": False,
                "points": [],
                "summary": ["Schools overlay unavailable: dataset returned no rows."],
                "loaded": True,
            }

        if "school_latitude" not in df.columns or "school_longitude" not in df.columns:
            return {
                "available": False,
                "points": [],
                "summary": ["Schools overlay unavailable: school_latitude/school_longitude columns were not found."],
                "loaded": True,
            }

        df["latitude"] = pd.to_numeric(df["school_latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["school_longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        df = df[df.apply(lambda r: _in_chicago_bounds(float(r["latitude"]), float(r["longitude"])), axis=1)]

        df["school_type"] = df.get("school_type", "Unknown").fillna("Unknown").astype(str)
        df["school_year"] = df.get("school_year", "Unknown").fillna("Unknown").astype(str)

        if school_type != "all":
            df = df[df["school_type"] == school_type]
        if school_year != "all":
            df = df[df["school_year"] == school_year]

        metric_type = "numeric"
        if metric == "school_type":
            metric_type = "categorical"
            categories = sorted(df["school_type"].dropna().unique().tolist())
            category_to_idx = {name: idx for idx, name in enumerate(categories)}
            df["metric_value"] = df["school_type"].map(category_to_idx).astype(float)
        else:
            if metric not in df.columns:
                metric = "graduation_rate_mean"
            df["metric_value"] = pd.to_numeric(df.get(metric), errors="coerce")
            df = df.dropna(subset=["metric_value"])

        points = []
        for row in df.itertuples(index=False):
            points.append(
                {
                    "name": getattr(row, "short_name", None) or getattr(row, "long_name", None) or "School",
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                    "school_type": str(getattr(row, "school_type", "Unknown")),
                    "school_year": str(getattr(row, "school_year", "Unknown")),
                    "metric": metric,
                    "value": float(row.metric_value),
                    "graduation_rate_mean": _safe_float(getattr(row, "graduation_rate_mean", None)),
                }
            )

        summary = [
            f"Dataset: Chicago Public Schools - School Profile Information SY2425 ({SCHOOLS_DATASET_ID})",
            f"Metric: {metric}",
            f"School type filter: {school_type}",
            f"School year filter: {school_year}",
            f"Points loaded: {len(points)}",
        ]

        return {
            "available": bool(points),
            "points": points,
            "summary": summary,
            "loaded": True,
            "metric": metric,
            "metric_type": metric_type,
        }
    except Exception as exc:  # pragma: no cover - network safety
        return {
            "available": False,
            "points": [],
            "summary": [f"Schools overlay unavailable: {exc}"],
            "loaded": True,
            "metric": metric,
        }


@lru_cache(maxsize=8)
def _census_tract_centroids(year: int) -> pd.DataFrame:
    if gpd is None:
        raise RuntimeError("geopandas is required for census tract geometry support")

    tracts_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_17_tract.zip"
    gdf = gpd.read_file(tracts_url)
    gdf = gdf[gdf["COUNTYFP"] == "031"].copy()
    gdf["tract"] = gdf["TRACTCE"].astype(str).str.zfill(6)

    # Compute centroids in a projected CRS, then convert to WGS84 for mapping.
    gdf_projected = gdf.to_crs(epsg=26916)
    centroid_geoms = gpd.GeoSeries(gdf_projected.geometry.centroid, crs=gdf_projected.crs).to_crs(epsg=4326)

    centroid_df = pd.DataFrame(
        {
            "tract": gdf["tract"].to_list(),
            "latitude": [geom.y for geom in centroid_geoms],
            "longitude": [geom.x for geom in centroid_geoms],
        }
    )
    centroid_df = centroid_df[
        centroid_df.apply(lambda r: _in_chicago_bounds(float(r["latitude"]), float(r["longitude"])), axis=1)
    ]
    return centroid_df.reset_index(drop=True)


def fetch_census_overlay(metric: str = "black_share", census_year: int = DEFAULT_CENSUS_YEAR) -> dict[str, Any]:
    try:
        fields = ["B01003_001E", "B02001_002E", "B02001_003E", "B02001_005E", "B19013_001E"]
        params = {
            "get": "NAME," + ",".join(fields),
            "for": "tract:*",
            "in": "state:17 county:031",
        }
        base_url = f"https://api.census.gov/data/{census_year}/acs/acs5"
        url = f"{base_url}?{urlencode(params)}"
        rows = _request_json(url)
        census_df = pd.DataFrame(rows[1:], columns=rows[0])
        for col in fields:
            census_df[col] = pd.to_numeric(census_df[col], errors="coerce")

        census_df["tract"] = census_df["tract"].astype(str).str.zfill(6)
        census_df = census_df.dropna(subset=["B01003_001E"])
        census_df = census_df[census_df["B01003_001E"] > 0]
        census_df["white_population"] = census_df["B02001_002E"]
        census_df["black_share"] = 100.0 * census_df["B02001_003E"] / census_df["B01003_001E"]
        census_df["asian_share"] = 100.0 * census_df["B02001_005E"] / census_df["B01003_001E"]
        census_df["median_household_income"] = census_df["B19013_001E"]

        centroid_df = _census_tract_centroids(census_year)
        merged = centroid_df.merge(census_df, on="tract", how="inner")

        if metric not in {"white_population", "black_share", "asian_share", "median_household_income"}:
            metric = "black_share"

        merged["metric_value"] = pd.to_numeric(merged[metric], errors="coerce")
        merged = merged.dropna(subset=["metric_value"])

        points = []
        for row in merged.itertuples(index=False):
            points.append(
                {
                    "tract": row.tract,
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                    "metric": metric,
                    "value": float(row.metric_value),
                    "population": int(row.B01003_001E),
                }
            )

        metric_labels = {
            "white_population": "White Population (tract count)",
            "black_share": "Black Share of Tract",
            "asian_share": "Asian Share of Tract",
            "median_household_income": "Median Household Income",
        }
        metric_label = metric_labels.get(metric, metric)
        summary = [
            f"Dataset: ACS 5-year ({census_year})",
            f"Metric: {metric_label}",
            f"Tracts loaded: {len(points)}",
        ]
        if points:
            values = [p["value"] for p in points]
            mean_value = sum(values) / len(values)
            if metric in {"black_share", "asian_share"}:
                summary.append(f"Mean share: {mean_value:.2f}%")
            elif metric == "median_household_income":
                summary.append(f"Mean tract median income: ${mean_value:,.0f}")
            else:
                summary.append(f"Average white population: {mean_value:,.0f}")

        return {
            "available": bool(points),
            "points": points,
            "summary": summary,
            "loaded": True,
            "metric": metric,
            "census_year": int(census_year),
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "points": [],
            "summary": [f"Census overlay unavailable: {exc}"],
            "loaded": True,
            "metric": metric,
            "census_year": int(census_year),
        }
