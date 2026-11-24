"""Flask entry point for the Chicago AQI Sensor Network experience."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import os

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sodapy import Socrata

try:
    import osmnx as ox
except Exception:  # pragma: no cover - optional dependency
    ox = None

from core import config, geo_utils, weather_fetcher
import data_intake
import interpolation_models


app = Flask(__name__)
DEFAULT_MAP_CENTER = {"lat": 41.8781, "lon": -87.6298}
STADIA_MAPS_API_KEY_ENV_VAR = "STADIA_MAPS_API_KEY"


def _runtime_state_from_prepared(prepared: data_intake.PreparedData) -> tuple[dict[str, float], geo_utils.BoundingBox]:
    catalog = prepared.sensor_catalog.copy()
    if "latitude" not in catalog.columns or "longitude" not in catalog.columns:
        return (
            dict(DEFAULT_MAP_CENTER),
            geo_utils.BoundingBox(41.5, 42.2, -88.0, -87.2),
        )

    catalog["latitude"] = pd.to_numeric(catalog["latitude"], errors="coerce")
    catalog["longitude"] = pd.to_numeric(catalog["longitude"], errors="coerce")
    catalog = catalog.dropna(subset=["latitude", "longitude"])

    if catalog.empty:
        return (
            dict(DEFAULT_MAP_CENTER),
            geo_utils.BoundingBox(41.5, 42.2, -88.0, -87.2),
        )

    map_center = {
        "lat": float(catalog["latitude"].mean()),
        "lon": float(catalog["longitude"].mean()),
    }
    bbox = geo_utils.BoundingBox.from_points(
        catalog["latitude"],
        catalog["longitude"],
        padding=0.1,
    )
    return map_center, bbox


def _reload_air_data(refresh: bool) -> data_intake.PreparedData:
    global PREPARED, MAP_CENTER, WORKSPACE_BBOX

    prepared = data_intake.get_prepared_data(refresh=refresh)
    map_center, bbox = _runtime_state_from_prepared(prepared)
    PREPARED = prepared
    MAP_CENTER = map_center
    WORKSPACE_BBOX = bbox

    aggregated_fn = globals().get("get_aggregated")
    if aggregated_fn is not None:
        aggregated_fn.cache_clear()

    return PREPARED


def _empty_prepared_data() -> data_intake.PreparedData:
    return data_intake.PreparedData(
        time_series=pd.DataFrame(columns=["sensor_name", "time", "latitude", "longitude", "aqi_value"]),
        sensor_catalog=pd.DataFrame(columns=["sensor_name", "latitude", "longitude"]),
        source="null",
        refreshed_at=None,
        source_warning=None,
    )


def _ensure_air_data_loaded(refresh: bool = False) -> data_intake.PreparedData:
    global PREPARED, MAP_CENTER, WORKSPACE_BBOX, AIR_DATA_LOAD_ERROR

    should_load = refresh or PREPARED.time_series.empty
    if not should_load:
        return PREPARED

    try:
        AIR_DATA_LOAD_ERROR = None
        return _reload_air_data(refresh=refresh)
    except Exception as exc:
        PREPARED = _empty_prepared_data()
        MAP_CENTER = dict(DEFAULT_MAP_CENTER)
        WORKSPACE_BBOX = geo_utils.BoundingBox(41.5, 42.2, -88.0, -87.2)
        AIR_DATA_LOAD_ERROR = str(exc)
        aggregated_fn = globals().get("get_aggregated")
        if aggregated_fn is not None:
            aggregated_fn.cache_clear()
        return PREPARED


def _provided_refresh_secret() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get(config.REFRESH_SECRET_HEADER, "").strip()


def _workspace_mapbox_config() -> dict[str, object]:
    stadia_key = os.getenv(STADIA_MAPS_API_KEY_ENV_VAR, "").strip()
    if not stadia_key:
        return {"style": "carto-darkmatter", "layers": []}
    tile_url = f"https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{{z}}/{{x}}/{{y}}.png?api_key={stadia_key}"
    return {
        "style": "white-bg",
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "source": [tile_url],
                "sourceattribution": "© Stadia Maps © OpenMapTiles © OpenStreetMap contributors",
            }
        ],
    }


PREPARED = _empty_prepared_data()
MAP_CENTER = dict(DEFAULT_MAP_CENTER)
WORKSPACE_BBOX = geo_utils.BoundingBox(41.5, 42.2, -88.0, -87.2)
AIR_DATA_LOAD_ERROR: str | None = None
WORKSPACE_METHOD_OPTIONS = [("Linear", "linear"), ("IDW", "idw"), ("Kriging", "kriging")]
STREET_MAP_PLACE_NAME = "Chicago, Illinois, USA"
CHICAGO_OPENDATA_DOMAIN = "data.cityofchicago.org"
CENSUS_YEAR = 2022

ENERGY_DATASET_IDS = {
    "tepd-j7h5": "Chicago Energy Benchmarking - 2014 Data Reported in 2015",
    "ebtp-548e": "Chicago Energy Benchmarking - 2015 Data Reported in 2016",
    "fpwt-snya": "Chicago Energy Benchmarking - 2016 Data Reported in 2017",
    "j2ev-2azp": "Chicago Energy Benchmarking - 2017 Data Reported in 2018",
    "m2kv-bmi3": "Chicago Energy Benchmarking - 2018 Data Reported in 2019",
    "jn94-it7m": "Chicago Energy Benchmarking - 2019 Data Reported in 2020",
    "ydbk-8hi6": "Chicago Energy Benchmarking - 2020 Data Reported in 2021",
    "gkf4-txtp": "Chicago Energy Benchmarking - 2021 Data Reported in 2022",
    "mz3g-jagv": "Chicago Energy Benchmarking - 2022 Data Reported in 2023",
    "3a36-5x9a": "Chicago Energy Benchmarking - 2023 Data Reported in 2024",
    "g5i5-yz37": "Chicago Energy Benchmarking - Covered Buildings",
}
DEFAULT_ENERGY_DATASET_ID = "3a36-5x9a"
DEFAULT_ENERGY_METRIC = "site_eui_kbtu_sq_ft"
ENERGY_NUMERIC_FIELDS = [
    "property_gross_floor_area_epa_calculated_buildings_sq_ft",
    "year_built",
    "number_of_buildings",
    "electricity_use_grid_purchase_and_generated_from_onsite_renewable_systems_kbtu",
    "natural_gas_use_kbtu",
    "site_eui_kbtu_sq_ft",
    "source_eui_kbtu_sq_ft",
    "weather_normalized_site_eui_kbtu_sq_ft",
    "weather_normalized_source_eui_kbtu_sq_ft",
    "total_ghg_emissions_metric_tons_co2e",
    "ghg_intensity_metric_tons_co2e_sq_ft",
    "energy_star_score",
    "district_chilled_water_use_kbtu",
    "district_steam_use_kbtu",
    "all_other_fuel_use_kbtu",
]

CENSUS_METRICS = {
    "median_income": {
        "variable": "B19013_001E",
        "label": "Median HH Income",
        "colorscale": "YlOrRd",
        "is_percent": False,
    },
    "white_pop": {
        "variable": "white_share",
        "label": "White Share of Tract",
        "colorscale": "Viridis",
        "is_percent": True,
    },
    "black_pop": {
        "variable": "black_share",
        "label": "Black Share of Tract",
        "colorscale": "Viridis",
        "is_percent": True,
    },
    "asian_pop": {
        "variable": "asian_share",
        "label": "Asian Share of Tract",
        "colorscale": "Viridis",
        "is_percent": True,
    },
}
CENSUS_YEAR_OPTIONS = [2023, 2022, 2021, 2020, 2019, 2018, 2017]
DEFAULT_CENSUS_VIEW = "single"
DEFAULT_CENSUS_YEAR_NEW = CENSUS_YEAR_OPTIONS[0]
DEFAULT_CENSUS_YEAR_OLD = CENSUS_YEAR_OPTIONS[1] if len(CENSUS_YEAR_OPTIONS) > 1 else CENSUS_YEAR_OPTIONS[0]
AIR_QUALITY_SUBVIEWS = {"sensors", "network", "interpolation"}
DEFAULT_AIR_QUALITY_SUBVIEW = "sensors"
AIR_METRIC_LABELS = {
    "aqi_value": "PM2.5 Mass (Configured)",
    "pm2_5concmassindividual_raw": "PM2.5 Mass (Raw)",
    "pm2_5concnumindividual_raw": "PM2.5 Number (Raw)",
    "pm2_5concnumindividual_value": "PM2.5 Number (Calibrated)",
    "no2concindividual_raw": "NO2 (Raw)",
    "no2concindividual_value": "NO2 (Calibrated)",
    "atmpressureindividual_raw": "Atmospheric Pressure (Raw)",
    "atmpressureindividual_value": "Atmospheric Pressure (Calibrated)",
    "relhumidambientindividual": "Ambient Relative Humidity",
    "temperatureambientindividual": "Ambient Temperature",
    "windspeedindividual_value": "Wind Speed",
    "winddirectionindividual_value": "Wind Direction",
}


def _air_metric_label(column_name: str) -> str:
    if column_name in AIR_METRIC_LABELS:
        return AIR_METRIC_LABELS[column_name]
    return column_name.replace("_", " ").title()


def _air_metric_options(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty:
        return [{"value": "aqi_value", "label": _air_metric_label("aqi_value")}]

    excluded_columns = {"latitude", "longitude"}
    preferred_order = [
        "aqi_value",
        "pm2_5concmassindividual_raw",
        "pm2_5concnumindividual_raw",
        "pm2_5concnumindividual_value",
        "no2concindividual_raw",
        "no2concindividual_value",
        "atmpressureindividual_raw",
        "atmpressureindividual_value",
        "relhumidambientindividual",
        "temperatureambientindividual",
        "windspeedindividual_value",
        "winddirectionindividual_value",
    ]

    valid: list[str] = []
    for col in df.columns:
        if col in excluded_columns or col.startswith(":@computed_region_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().sum() == 0:
            continue
        valid.append(col)

    ordered: list[str] = [col for col in preferred_order if col in valid]
    ordered.extend(col for col in valid if col not in set(ordered))

    if "aqi_value" not in ordered and "aqi_value" in df.columns:
        ordered.insert(0, "aqi_value")

    if not ordered:
        ordered = ["aqi_value"]

    return [{"value": col, "label": _air_metric_label(col)} for col in ordered]


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _normalized_coverage_threshold(value: str | None, default: float) -> float:
    threshold = _safe_float(value, default)
    # Accept both fractional input (0.5) and percentage-style input (50).
    if threshold > 1.0:
        threshold = threshold / 100.0
    return min(max(threshold, 0.0), 1.0)


@lru_cache(maxsize=1)
def _open_data_client() -> Socrata:
    return Socrata(CHICAGO_OPENDATA_DOMAIN, None)


def _fetch_all_socrata_rows(api_key: str, page_size: int = 50_000) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    client = _open_data_client()
    while True:
        batch = client.get(api_key, limit=page_size, offset=offset)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


@lru_cache(maxsize=len(ENERGY_DATASET_IDS))
def _energy_dataset_frame(dataset_id: str) -> pd.DataFrame:
    records = _fetch_all_socrata_rows(dataset_id)
    return pd.DataFrame.from_records(records)


def _normalize_energy_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "location" in work.columns and ("latitude" not in work.columns or "longitude" not in work.columns):
        lat_vals: list[float] = []
        lon_vals: list[float] = []
        for val in work["location"].tolist():
            if isinstance(val, dict):
                lat_vals.append(val.get("latitude"))
                lon_vals.append(val.get("longitude"))
            else:
                lat_vals.append(np.nan)
                lon_vals.append(np.nan)
        if "latitude" not in work.columns:
            work["latitude"] = lat_vals
        if "longitude" not in work.columns:
            work["longitude"] = lon_vals

    for col in ENERGY_NUMERIC_FIELDS + ["latitude", "longitude"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "latitude" in work.columns and "longitude" in work.columns:
        work = work.dropna(subset=["latitude", "longitude"])
        work = work[(work["latitude"].between(41.55, 42.10)) & (work["longitude"].between(-88.0, -87.2))]

    return work


def _energy_metric_options(df: pd.DataFrame) -> list[dict[str, str]]:
    options = []
    for metric in ENERGY_NUMERIC_FIELDS:
        if metric not in df.columns:
            continue
        col = pd.to_numeric(df[metric], errors="coerce")
        if col.notna().sum() == 0:
            continue
        options.append({"value": metric, "label": metric.replace("_", " ").title()})
    return options


def _build_energy_overlay(dataset_id: str, metric: str) -> dict[str, object]:
    if dataset_id not in ENERGY_DATASET_IDS:
        dataset_id = DEFAULT_ENERGY_DATASET_ID

    dataset_name = ENERGY_DATASET_IDS[dataset_id]
    try:
        df_raw = _energy_dataset_frame(dataset_id)
    except Exception as exc:
        return {
            "available": False,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "metric": metric,
            "metric_label": metric,
            "metric_options": [],
            "points": [],
            "summary": [f"Energy overlay unavailable: {exc}"],
            "loaded": True,
        }

    df = _normalize_energy_df(df_raw)
    metric_options = _energy_metric_options(df)
    metric_values = {opt["value"] for opt in metric_options}
    if metric not in metric_values:
        metric = DEFAULT_ENERGY_METRIC if DEFAULT_ENERGY_METRIC in metric_values else (metric_options[0]["value"] if metric_options else metric)

    if df.empty or metric not in df.columns:
        return {
            "available": False,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "metric": metric,
            "metric_label": metric.replace("_", " ").title(),
            "metric_options": metric_options,
            "points": [],
            "summary": ["No mappable rows available for selected energy dataset."],
            "loaded": True,
        }

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric, "latitude", "longitude"])
    if df.empty:
        return {
            "available": False,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "metric": metric,
            "metric_label": metric.replace("_", " ").title(),
            "metric_options": metric_options,
            "points": [],
            "summary": ["All rows are missing coordinates or selected metric values."],
            "loaded": True,
        }

    if df.shape[0] > 12_000:
        df = df.sample(n=12_000, random_state=7)

    points = []
    for row in df.itertuples(index=False):
        property_type = getattr(row, "primary_property_type_epa_calculated", "n/a")
        community_area = getattr(row, "community_area", "n/a")
        points.append(
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "value": float(getattr(row, metric)),
                "property_type": str(property_type) if property_type is not None else "n/a",
                "community_area": str(community_area) if community_area is not None else "n/a",
            }
        )

    series = pd.to_numeric(df[metric], errors="coerce")
    metric_label = metric.replace("_", " ").title()
    summary = [
        f"Dataset: {dataset_name}",
        f"Rows loaded: {len(points):,}",
        f"Metric: {metric_label}",
        f"Mean: {series.mean():.2f}",
        f"Median: {series.median():.2f}",
    ]

    return {
        "available": True,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "metric": metric,
        "metric_label": metric_label,
        "metric_options": metric_options,
        "points": points,
        "summary": summary,
        "loaded": True,
    }


def _tract_id_from_properties(properties: dict[str, object]) -> str:
    if not properties:
        return ""
    for key in ["GEOID", "geoid", "GEOID10", "GEOID20"]:
        value = properties.get(key)
        if value:
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if len(digits) >= 11:
                return digits[-11:]
            return digits.zfill(11)

    state = str(properties.get("STATE") or properties.get("STATEFP") or "")
    county = str(properties.get("COUNTY") or properties.get("COUNTYFP") or "")
    tract_raw = str(properties.get("TRACT") or properties.get("TRACTCE") or "")
    tract_digits = "".join(ch for ch in tract_raw if ch.isdigit())
    tract = tract_digits[:6] if len(tract_digits) >= 6 else tract_digits.zfill(6)
    if state and county and tract:
        state_digits = "".join(ch for ch in state if ch.isdigit()).zfill(2)
        county_digits = "".join(ch for ch in county if ch.isdigit()).zfill(3)
        return f"{state_digits}{county_digits}{tract}"
    return ""


@lru_cache(maxsize=1)
def _chicago_tract_geojson() -> dict[str, object]:
    # TIGERWeb geojson avoids adding a geopandas runtime dependency to the Flask app.
    base_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/4/query"
    params = {
        "where": "STATE = '17' AND COUNTY = '031'",
        "outFields": "GEOID,STATE,COUNTY,TRACT",
        "outSR": 4326,
        "f": "geojson",
    }
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    if isinstance(raw, dict) and raw.get("error"):
        message = raw["error"].get("message") if isinstance(raw["error"], dict) else str(raw["error"])
        raise RuntimeError(f"TIGERWeb request failed: {message}")
    features = []
    for feature in raw.get("features", []):
        properties = feature.get("properties", {})
        geoid = _tract_id_from_properties(properties)
        if not geoid.startswith("17031"):
            continue
        properties["GEOID"] = geoid
        feature["properties"] = properties
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


@lru_cache(maxsize=8)
def _census_tract_frame(year: int) -> pd.DataFrame:
    selected_vars = [
        "B01003_001E",
        "B19013_001E",
        "B02001_002E",
        "B02001_003E",
        "B02001_005E",
    ]
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(selected_vars),
        "for": "tract:*",
        "in": "state:17 county:031",
    }
    census_key = os.getenv(config.CENSUS_API_KEY_ENV_VAR, "").strip()
    if census_key:
        params["key"] = census_key

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "json" not in content_type.lower():
        snippet = resp.text[:180].replace("\n", " ").strip()
        if "Missing Key" in snippet or "invalid key" in snippet.lower():
            raise RuntimeError(
                f"Census API key required. Set {config.CENSUS_API_KEY_ENV_VAR} in your environment."
            )
        raise RuntimeError(f"Census API returned non-JSON content: {snippet}")

    rows = resp.json()
    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError("Census API returned no tract data for Cook County, IL.")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for var in selected_vars:
        df[var] = pd.to_numeric(df[var], errors="coerce")
    df.loc[df["B19013_001E"] < 0, "B19013_001E"] = pd.NA
    df["GEOID"] = "17" + df["county"].astype(str).str.zfill(3) + df["tract"].astype(str).str.zfill(6)
    df["white_share"] = df["B02001_002E"] / df["B01003_001E"]
    df["black_share"] = df["B02001_003E"] / df["B01003_001E"]
    df["asian_share"] = df["B02001_005E"] / df["B01003_001E"]
    return df


def _build_census_overlay(metric_key: str, view_mode: str, year_new: int, year_old: int) -> dict[str, object]:
    if metric_key not in CENSUS_METRICS:
        metric_key = "median_income"
    if view_mode not in {"single", "change"}:
        view_mode = DEFAULT_CENSUS_VIEW
    valid_years = set(CENSUS_YEAR_OPTIONS)
    if year_new not in valid_years:
        year_new = DEFAULT_CENSUS_YEAR_NEW
    if year_old not in valid_years:
        year_old = DEFAULT_CENSUS_YEAR_OLD

    metric_meta = CENSUS_METRICS[metric_key]
    metric_var = metric_meta["variable"]

    try:
        geojson = deepcopy(_chicago_tract_geojson())
        new_df = _census_tract_frame(year_new)
        old_df = _census_tract_frame(year_old) if view_mode == "change" else None
    except Exception as exc:
        return {
            "available": False,
            "metric": metric_key,
            "metric_label": metric_meta["label"],
            "colorscale": metric_meta["colorscale"],
            "is_percent": metric_meta["is_percent"],
            "geojson": {"type": "FeatureCollection", "features": []},
            "locations": [],
            "z": [],
            "hover": [],
            "view": view_mode,
            "year_new": year_new,
            "year_old": year_old,
            "summary": [f"Census overlay unavailable: {exc}"],
            "loaded": True,
        }

    new_map = new_df.set_index("GEOID")[metric_var].to_dict()
    old_map = old_df.set_index("GEOID")[metric_var].to_dict() if old_df is not None else {}

    locations: list[str] = []
    z_values: list[float | None] = []
    hover_text: list[str] = []
    zmid = None
    colorscale = metric_meta["colorscale"]

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geoid = props.get("GEOID") or _tract_id_from_properties(props)
        if not geoid:
            continue

        value_new = new_map.get(str(geoid))
        value_old = old_map.get(str(geoid)) if view_mode == "change" else None

        if pd.isna(value_new):
            value_new = None
        if view_mode == "change" and pd.isna(value_old):
            value_old = None

        if view_mode == "change":
            value = None if (value_new is None or value_old is None) else float(value_new) - float(value_old)
        else:
            value = value_new

        locations.append(str(geoid))
        z_values.append(None if value is None else float(value))

        if view_mode == "change":
            if value is None:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new} - {year_old}): n/a")
            elif metric_meta["is_percent"]:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new} - {year_old}): {float(value) * 100:+.2f} pp")
            else:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new} - {year_old}): ${float(value):+,.0f}")
        else:
            if value is None:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new}): n/a")
            elif metric_meta["is_percent"]:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new}): {float(value):.2%}")
            else:
                hover_text.append(f"Tract {geoid}<br>{metric_meta['label']} ({year_new}): ${float(value):,.0f}")

    series = pd.Series(z_values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    matched_values = int(series.shape[0])
    if series.empty:
        summary = ["No census tract values were available for this metric."]
    elif view_mode == "change" and metric_meta["is_percent"]:
        colorscale = "RdBu"
        zmid = 0.0
        summary = [
            f"Metric: {metric_meta['label']} ({year_new} - {year_old})",
            f"Tracts loaded: {len(locations):,}",
            f"Tracts with values: {matched_values:,}",
            f"Mean change: {series.mean() * 100:+.2f} pp",
            f"Median change: {series.median() * 100:+.2f} pp",
        ]
    elif view_mode == "change":
        colorscale = "RdBu"
        zmid = 0.0
        summary = [
            f"Metric: {metric_meta['label']} ({year_new} - {year_old})",
            f"Tracts loaded: {len(locations):,}",
            f"Tracts with values: {matched_values:,}",
            f"Mean change: ${series.mean():+,.0f}",
            f"Median change: ${series.median():+,.0f}",
        ]
    elif metric_meta["is_percent"]:
        summary = [
            f"Metric: {metric_meta['label']} ({year_new})",
            f"Tracts loaded: {len(locations):,}",
            f"Tracts with values: {matched_values:,}",
            f"Mean: {series.mean():.2%}",
            f"Median: {series.median():.2%}",
        ]
    else:
        summary = [
            f"Metric: {metric_meta['label']} ({year_new})",
            f"Tracts loaded: {len(locations):,}",
            f"Tracts with values: {matched_values:,}",
            f"Mean: ${series.mean():,.0f}",
            f"Median: ${series.median():,.0f}",
        ]

    return {
        "available": bool(locations) and matched_values > 0,
        "metric": metric_key,
        "metric_label": f"{metric_meta['label']} ({year_new} - {year_old})" if view_mode == "change" else f"{metric_meta['label']} ({year_new})",
        "colorscale": colorscale,
        "is_percent": metric_meta["is_percent"],
        "view": view_mode,
        "year_new": year_new,
        "year_old": year_old,
        "zmid": zmid,
        "geojson": geojson,
        "locations": locations,
        "z": z_values,
        "hover": hover_text,
        "summary": summary,
        "loaded": True,
    }


@lru_cache(maxsize=128)
def get_aggregated(bin_code: str, metric_column: str = "aqi_value") -> pd.DataFrame:
    _ensure_air_data_loaded()
    source = PREPARED.time_series
    required_columns = {"sensor_name", "time", "latitude", "longitude"}
    if source.empty or not required_columns.issubset(source.columns):
        return pd.DataFrame(columns=["sensor_name", "time", "latitude", "longitude", "aqi_value", "reading_count"])
    value_column = metric_column if metric_column in source.columns else "aqi_value"
    if value_column not in source.columns:
        return pd.DataFrame(columns=["sensor_name", "time", "latitude", "longitude", "aqi_value", "reading_count"])
    df = source[["sensor_name", "time", "latitude", "longitude", value_column]].copy()
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df["time_bin"] = df["time"].dt.floor(bin_code)
    aggregated = (
        df.groupby(["sensor_name", "time_bin", "latitude", "longitude"])
        .agg(aqi_value=(value_column, "mean"), reading_count=(value_column, "count"))
        .reset_index()
        .rename(columns={"time_bin": "time"})
        .sort_values("time")
    )
    return aggregated


def _time_options(bin_code: str, min_coverage: float = 0.0, metric_column: str = "aqi_value") -> list[dict[str, str]]:
    df = get_aggregated(bin_code, metric_column)
    if df.empty:
        return []
    total_sensors = max(PREPARED.sensor_catalog.shape[0], 1)
    coverage = (
        df.groupby("time")["sensor_name"].nunique().reset_index(name="active_sensors")
    )
    coverage["coverage"] = coverage["active_sensors"] / total_sensors
    filtered = coverage[coverage["coverage"] >= min_coverage]
    if filtered.empty:
        return []
    times = pd.to_datetime(filtered["time"]).sort_values()
    return [{"label": ts.strftime("%Y-%m-%d %H:%M"), "value": ts.isoformat()} for ts in times]


def _uptime_stats(stats: dict[str, float]) -> list[str]:
    entries = [
        f"Observing: {stats['observing']}",
        f"Not observing: {stats['not_observing']}",
        f"Coverage: {stats['coverage_pct']:.1f} %",
        f"Range: {stats['range_km']:.1f} km",
        f"Mean distance: {stats['mean_km']:.1f} km",
        f"Min distance: {stats['min_km']:.1f} km",
    ]
    if stats.get("insufficient"):
        entries.append("Warning: coverage below threshold")
    return entries


def _air_data_status() -> dict[str, object]:
    _ensure_air_data_loaded()
    if PREPARED.time_series.empty or "time" not in PREPARED.time_series.columns:
        return {
            "label": "Unavailable",
            "tone": "warning",
            "latest_observation": None,
            "latest_age_hours": None,
            "refreshed_at": None,
            "warning": AIR_DATA_LOAD_ERROR,
        }

    latest_time = pd.to_datetime(PREPARED.time_series["time"], errors="coerce").max()
    now = datetime.now(timezone.utc)
    latest_age_hours = None
    if pd.notna(latest_time):
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize(timezone.utc)
        latest_age_hours = max(0.0, (now - latest_time.to_pydatetime()).total_seconds() / 3600.0)
    source = PREPARED.source
    if source == "live":
        label = "Live Socrata"
        tone = "good"
    elif source == "cache-fallback":
        label = "Cached fallback"
        tone = "warning"
    else:
        label = "Bundled cache"
        tone = "warning"
    return {
        "label": label,
        "tone": tone,
        "latest_observation": latest_time.isoformat() if pd.notna(latest_time) else None,
        "latest_age_hours": latest_age_hours,
        "refreshed_at": PREPARED.refreshed_at,
        "warning": PREPARED.source_warning,
    }


def _workspace_report(
    aggregated: pd.DataFrame,
    time_value: pd.Timestamp,
    uptime_stats: dict[str, float],
    metric_label: str,
) -> dict[str, object]:
    current = aggregated[aggregated["time"] == time_value].copy()
    values = pd.to_numeric(current["aqi_value"], errors="coerce").dropna()
    times = pd.to_datetime(aggregated.loc[aggregated["time"] <= time_value, "time"]).sort_values().drop_duplicates()
    previous_times = times[times < time_value]
    previous_time = previous_times.iloc[-1] if not previous_times.empty else pd.NaT
    previous = aggregated[aggregated["time"] == previous_time] if pd.notna(previous_time) else pd.DataFrame()
    previous_mean = pd.to_numeric(previous["aqi_value"], errors="coerce").mean() if not previous.empty else np.nan
    current_mean = float(values.mean()) if not values.empty else np.nan
    absolute_change = current_mean - previous_mean if not np.isnan(previous_mean) else np.nan
    percent_change = (
        absolute_change / previous_mean * 100.0
        if np.isfinite(previous_mean) and previous_mean != 0
        else np.nan
    )

    ranked = current.loc[current["aqi_value"].notna()].sort_values("aqi_value", ascending=False)
    highest = ranked.iloc[0] if not ranked.empty else None
    lowest = ranked.iloc[-1] if not ranked.empty else None

    trend = (
        aggregated.loc[aggregated["time"] <= time_value]
        .groupby("time")["aqi_value"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("time")
        .tail(18)
    )

    def number(value: float, digits: int = 1) -> str:
        return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"

    cards = [
        {"label": f"Average {metric_label}", "value": number(current_mean), "tone": "accent"},
        {
            "label": "vs. previous bin",
            "value": "n/a" if np.isnan(absolute_change) else f"{absolute_change:+.1f}",
            "detail": None if np.isnan(percent_change) else f"{percent_change:+.1f}%",
            "tone": "good" if absolute_change <= 0 else "warning",
        },
        {"label": "Sensor coverage", "value": f"{uptime_stats['coverage_pct']:.0f}%", "tone": "good" if uptime_stats["coverage_pct"] >= 75 else "warning"},
        {"label": "Active sensors", "value": f"{uptime_stats['observing']} / {uptime_stats['observing'] + uptime_stats['not_observing']}"},
    ]

    insights = []
    if highest is not None:
        insights.append(f"Highest reading: {highest['sensor_name']} at {float(highest['aqi_value']):.1f}.")
    if lowest is not None:
        insights.append(f"Lowest reading: {lowest['sensor_name']} at {float(lowest['aqi_value']):.1f}.")
    if np.isfinite(absolute_change):
        direction = "decreased" if absolute_change < 0 else "increased"
        insights.append(f"The citywide mean {direction} by {abs(absolute_change):.1f} from the previous bin.")
    if uptime_stats["not_observing"] > 0:
        insights.append(f"{uptime_stats['not_observing']} sensors had no valid reading in this bin.")

    return {
        "cards": cards,
        "insights": insights,
        "trend": [
            {
                "time": pd.Timestamp(row.time).isoformat(),
                "label": pd.Timestamp(row.time).strftime("%m/%d %H:%M"),
                "mean": float(row.mean),
                "count": int(row.count),
            }
            for row in trend.itertuples(index=False)
            if pd.notna(row.mean)
        ],
        "data_status": _air_data_status(),
        "metric_label": metric_label,
    }


def _network_overlay_summary(
    overlay: dict[str, list[dict[str, object]]],
    threshold_km: float,
) -> list[str]:
    nodes = overlay.get("nodes", [])
    edges = overlay.get("edges", [])
    observing_nodes = sum(1 for node in nodes if bool(node.get("is_observing")))
    edge_count = len(edges)

    if observing_nodes > 1:
        density = (2.0 * edge_count) / (observing_nodes * (observing_nodes - 1))
    else:
        density = 0.0

    avg_degree = (2.0 * edge_count / observing_nodes) if observing_nodes > 0 else 0.0

    return [
        f"Link threshold: {threshold_km:.2f} km",
        f"Displayed sensors: {len(nodes)}",
        f"Observing sensors: {observing_nodes}",
        f"Edges: {edge_count}",
        f"Density: {density:.3f}",
        f"Avg degree: {avg_degree:.2f}",
    ]


def _interpolation_stats(result: interpolation_models.InterpolationResult, metric_label: str) -> list[str]:
    return [
        f"Method: {result.method_used.upper()}",
        f"Metric: {metric_label}",
        f"Min: {result.grid_values.min():.1f}",
        f"Mean: {result.grid_values.mean():.1f}",
        f"Max: {result.grid_values.max():.1f}",
    ]


@lru_cache(maxsize=1)
def _build_street_overlay(place_name: str = STREET_MAP_PLACE_NAME) -> dict[str, object]:
    if ox is None:
        return {
            "available": False,
            "lat": [],
            "lon": [],
            "summary": ["Street map unavailable: install osmnx to enable this mode."],
        }

    try:
        graph = ox.graph_from_place(place_name, network_type="drive")
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    except Exception as exc:
        return {
            "available": False,
            "lat": [],
            "lon": [],
            "summary": [f"Street map unavailable: {exc}"],
        }

    lat: list[float | None] = []
    lon: list[float | None] = []
    edge_segments = 0

    for geom in edges.geometry:
        if geom is None:
            continue

        if geom.geom_type == "LineString":
            line_geometries = [geom]
        elif geom.geom_type == "MultiLineString":
            line_geometries = list(geom.geoms)
        else:
            continue

        for line in line_geometries:
            xs, ys = line.xy
            lon.extend(float(x) for x in xs)
            lat.extend(float(y) for y in ys)
            lon.append(None)
            lat.append(None)
            edge_segments += 1

    return {
        "available": True,
        "lat": lat,
        "lon": lon,
        "summary": [
            f"Street network: {place_name}",
            f"Drive graph nodes: {graph.number_of_nodes()}",
            f"Drive graph edges: {graph.number_of_edges()}",
            f"Rendered segments: {edge_segments}",
        ],
    }


@app.context_processor
def inject_nav_links():
    return {
        "nav_links": [
            ("Workspace", "map_workspace"),
        ]
    }


def _build_network_overlay(
    df_points: pd.DataFrame,
    show_all_points: bool,
    threshold_km: float,
) -> dict[str, list[dict[str, object]]]:
    nodes = []
    edges = []
    node_df = df_points[["sensor_name", "latitude", "longitude", "aqi_value", "is_observing"]].copy()
    node_df["aqi_value"] = node_df["aqi_value"].where(node_df["aqi_value"].notna(), None)

    for row in node_df.itertuples(index=False):
        is_observing = bool(row.is_observing)
        if not show_all_points and not is_observing:
            continue
        nodes.append(
            {
                "sensor_name": row.sensor_name,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "aqi_value": None if row.aqi_value is None else float(row.aqi_value),
                "is_observing": is_observing,
                "opacity": 0.28 if (show_all_points and not is_observing) else 0.9,
            }
        )

    obs = node_df[node_df["is_observing"]].reset_index(drop=True)
    if obs.shape[0] < 2:
        return {"nodes": nodes, "edges": edges}

    dist = geo_utils.pairwise_distance_matrix(obs)
    for i in range(obs.shape[0]):
        for j in range(i + 1, obs.shape[0]):
            d = float(dist[i, j])
            if np.isnan(d) or d > threshold_km:
                continue
            edges.append(
                {
                    "from": {
                        "lat": float(obs.loc[i, "latitude"]),
                        "lon": float(obs.loc[i, "longitude"]),
                    },
                    "to": {
                        "lat": float(obs.loc[j, "latitude"]),
                        "lon": float(obs.loc[j, "longitude"]),
                    },
                    "distance_km": d,
                }
            )
    return {"nodes": nodes, "edges": edges}


@app.route("/workspace")
def map_workspace():
    _ensure_air_data_loaded()
    json_request = request.args.get("format") == "json"
    bin_code = request.args.get("bin", config.DEFAULT_BIN)
    requested_mode = request.args.get("mode", "explorer")
    requested_aqi_view = request.args.get("aqi_view", DEFAULT_AIR_QUALITY_SUBVIEW)
    if requested_mode in {"network", "interpolation"}:
        requested_aqi_view = requested_mode
        requested_mode = "explorer"
    if requested_aqi_view not in AIR_QUALITY_SUBVIEWS:
        requested_aqi_view = DEFAULT_AIR_QUALITY_SUBVIEW
    needs_network_overlay = requested_aqi_view == "network"
    needs_interpolation_overlay = requested_aqi_view == "interpolation"

    air_metric_options = _air_metric_options(PREPARED.time_series)
    valid_air_metrics = {opt["value"] for opt in air_metric_options}
    selected_air_metric = request.args.get("aq_metric", "aqi_value")
    if selected_air_metric not in valid_air_metrics:
        selected_air_metric = air_metric_options[0]["value"]
    selected_air_metric_label = next(
        (opt["label"] for opt in air_metric_options if opt["value"] == selected_air_metric),
        _air_metric_label(selected_air_metric),
    )

    coverage_str = request.args.get("coverage", str(config.DEFAULT_COVERAGE_THRESHOLD))
    coverage_threshold = _normalized_coverage_threshold(coverage_str, config.DEFAULT_COVERAGE_THRESHOLD)
    method = request.args.get("method", "linear")
    valid_methods = {value for _, value in WORKSPACE_METHOD_OPTIONS}
    if method not in valid_methods:
        method = "linear"
    show_all_points = request.args.get("show_all", "0") in {"1", "true", "True", "on"}
    distance_limit_km = _safe_float(request.args.get("distance_km"), np.nan)

    grid_res_str = request.args.get("grid", str(config.DEFAULT_GRID_RESOLUTION))
    try:
        grid_resolution = int(grid_res_str)
    except ValueError:
        grid_resolution = config.DEFAULT_GRID_RESOLUTION
    grid_resolution = int(np.clip(grid_resolution, 30, 120))

    time_options = _time_options(bin_code, coverage_threshold, selected_air_metric)
    if not time_options:
        if json_request:
            return jsonify(
                {
                    "workspace_payload": {
                        "map_center": MAP_CENTER,
                        "mapbox_config": _workspace_mapbox_config(),
                        "report": {"cards": [], "insights": [], "trend": [], "data_status": _air_data_status()},
                    },
                    "time_options": [],
                    "selected_time": None,
                    "distance_options_km": [],
                    "selected_distance_km": config.ADJACENCY_DISTANCE_THRESHOLD_KM,
                    "message": "No time bins meet the current coverage threshold.",
                }
            )
        return render_template(
            "workspace.html",
            full_bleed=True,
            bin_code=bin_code,
            bin_options=config.BIN_OPTIONS,
            coverage_threshold=coverage_threshold,
            coverage_options=config.COVERAGE_THRESHOLDS,
            method=method,
            method_options=WORKSPACE_METHOD_OPTIONS,
            show_all_points=show_all_points,
            distance_options_km=[],
            selected_distance_km=config.ADJACENCY_DISTANCE_THRESHOLD_KM,
            grid_resolution=grid_resolution,
            time_options=[],
            selected_time=None,
            workspace_payload={},
            message="No time bins meet the current coverage threshold.",
        )

    selected_time = request.args.get("time")
    if not selected_time or all(opt["value"] != selected_time for opt in time_options):
        selected_time = time_options[0]["value"]

    aggregated = get_aggregated(bin_code, selected_air_metric)
    time_value = pd.to_datetime(selected_time)
    merged, uptime_stats = data_intake.compute_uptime(
        aggregated,
        PREPARED.sensor_catalog,
        time_value,
        coverage_threshold,
    )
    distance_options_km: list[float] = []
    selected_distance_km = config.ADJACENCY_DISTANCE_THRESHOLD_KM
    if needs_network_overlay:
        distance_source = merged.copy() if show_all_points else merged[merged["is_observing"]].copy()
        distance_source = distance_source.dropna(subset=["latitude", "longitude"])
        condensed_dist = geo_utils.haversine_matrix(distance_source)
        if condensed_dist.size > 0:
            positive_dist = condensed_dist[condensed_dist > 0]
            if positive_dist.size == 0:
                positive_dist = np.array([1.0])

            dist_min = float(np.min(positive_dist))
            dist_max = float(np.max(positive_dist))
            if np.isclose(dist_min, dist_max):
                distance_options_km = [round(dist_min, 2)]
            else:
                raw_options = np.array([dist_max * 0.05, dist_max * 0.10, dist_max * 0.15, dist_max * 0.25, dist_max * 0.5, dist_max])
                clipped = np.clip(raw_options, dist_min, dist_max)
                unique_sorted = np.unique(np.round(clipped, 2))
                distance_options_km = [float(x) for x in unique_sorted if x > 0]
                if not distance_options_km:
                    distance_options_km = [round(max(dist_min, 0.1), 2), round(dist_max, 2)]

            if np.isnan(distance_limit_km):
                selected_distance_km = distance_options_km[min(2, len(distance_options_km) - 1)]
            else:
                selected_distance_km = min(distance_options_km, key=lambda x: abs(x - distance_limit_km))

    uptime_points = []
    for row in merged.itertuples(index=False):
        row_reading_count = getattr(row, "reading_count", np.nan)
        obs_count = 0 if pd.isna(row_reading_count) else int(row_reading_count)
        is_observing = obs_count > 0
        if not show_all_points and not is_observing:
            continue
        uptime_points.append(
            {
                "sensor_name": row.sensor_name,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "aqi_value": None if pd.isna(row.aqi_value) else float(row.aqi_value),
                "is_observing": is_observing,
                "observation_count": obs_count,
                "opacity": 0.22 if (show_all_points and not is_observing) else 0.95,
            }
        )

    df_slice = data_intake.filter_by_time(aggregated, time_value)

    network_overlay: dict[str, list[dict[str, object]]] = {"nodes": [], "edges": []}
    if needs_network_overlay:
        network_overlay = _build_network_overlay(
            merged,
            show_all_points=show_all_points,
            threshold_km=selected_distance_km,
        )

    interpolation_result = None
    if needs_interpolation_overlay:
        interpolation_result = interpolation_models.interpolate_time_slice(
            df_slice,
            method=method,
            grid_resolution=grid_resolution,
            bbox=WORKSPACE_BBOX,
        )

    interpolation_points: list[dict[str, float]] = []
    interpolation_stats: list[str] = []
    if interpolation_result is not None:
        flat_lon = interpolation_result.grid_lon.ravel()
        flat_lat = interpolation_result.grid_lat.ravel()
        flat_val = interpolation_result.grid_values.ravel()
        for lon, lat, value in zip(flat_lon, flat_lat, flat_val, strict=False):
            if np.isnan(value):
                continue
            interpolation_points.append(
                {
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "value": float(value),
                }
            )
        interpolation_stats = _interpolation_stats(interpolation_result, selected_air_metric_label)

    workspace_payload = {
        "map_center": MAP_CENTER,
        "mapbox_config": _workspace_mapbox_config(),
        "time": time_value.strftime("%Y-%m-%d %H:%M"),
        "active_mode": requested_mode,
        "active_aqi_view": requested_aqi_view,
        "show_all_points": show_all_points,
        "selected_distance_km": selected_distance_km,
        "distance_options_km": distance_options_km,
        "uptime_points": uptime_points,
        "uptime_stats": [f"Metric: {selected_air_metric_label}"] + _uptime_stats(uptime_stats),
        "network_overlay": network_overlay,
        "network_summary": [f"Metric: {selected_air_metric_label}"] + _network_overlay_summary(network_overlay, selected_distance_km),
        "interpolation_points": interpolation_points,
        "interpolation_stats": interpolation_stats,
        "interpolation_method_used": interpolation_result.method_used if interpolation_result else method,
        "report": _workspace_report(aggregated, time_value, uptime_stats, selected_air_metric_label),
        "air_config": {
            "selected_metric": selected_air_metric,
            "selected_metric_label": selected_air_metric_label,
            "metric_options": air_metric_options,
        },
        "street_overlay": {
            "available": False,
            "lat": [],
            "lon": [],
            "loaded": False,
        },
        "street_summary": ["Street network loads on demand. Click Streetmaps to fetch it."],
        "weather_overlay": {
            "available": False,
            "points": [],
            "loaded": False,
        },
        "weather_summary": ["Live NOAA weather loads on demand. Click Weather to fetch it."],
        "energy_overlay": {
            "available": False,
            "points": [],
            "loaded": False,
        },
        "energy_summary": ["Energy benchmarking points load on demand. Click Energy to fetch it."],
        "energy_config": {
            "dataset_options": [
                {"value": dataset_id, "label": dataset_name}
                for dataset_id, dataset_name in ENERGY_DATASET_IDS.items()
            ],
            "selected_dataset_id": DEFAULT_ENERGY_DATASET_ID,
            "selected_metric": DEFAULT_ENERGY_METRIC,
            "metric_options": [{"value": DEFAULT_ENERGY_METRIC, "label": "Site Eui Kbtu Sq Ft"}],
        },
        "census_overlay": {
            "available": False,
            "geojson": {"type": "FeatureCollection", "features": []},
            "locations": [],
            "z": [],
            "hover": [],
            "loaded": False,
        },
        "census_summary": ["Census tract choropleth loads on demand. Click Census to fetch it."],
        "census_config": {
            "selected_metric": "median_income",
            "selected_view": DEFAULT_CENSUS_VIEW,
            "selected_year_new": DEFAULT_CENSUS_YEAR_NEW,
            "selected_year_old": DEFAULT_CENSUS_YEAR_OLD,
            "view_options": [
                {"value": "single", "label": "Single Year"},
                {"value": "change", "label": "Change (New - Old)"},
            ],
            "year_options": [
                {"value": year, "label": str(year)}
                for year in CENSUS_YEAR_OPTIONS
            ],
            "metric_options": [
                {"value": key, "label": value["label"]}
                for key, value in CENSUS_METRICS.items()
            ],
        },
        "metric_range": {
            "min": float(np.nanmin(df_slice["aqi_value"].to_numpy())) if len(df_slice) else 0.0,
            "max": float(np.nanmax(df_slice["aqi_value"].to_numpy())) if len(df_slice) else 1.0,
        },
    }

    if json_request:
        return jsonify(
            {
                "workspace_payload": workspace_payload,
                "time_options": time_options,
                "selected_time": selected_time,
                "distance_options_km": distance_options_km,
                "selected_distance_km": selected_distance_km,
                "message": None,
            }
        )

    return render_template(
        "workspace.html",
        full_bleed=True,
        bin_code=bin_code,
        bin_options=config.BIN_OPTIONS,
        coverage_threshold=coverage_threshold,
        coverage_options=config.COVERAGE_THRESHOLDS,
        method=method,
        method_options=WORKSPACE_METHOD_OPTIONS,
        show_all_points=show_all_points,
        distance_options_km=distance_options_km,
        selected_distance_km=selected_distance_km,
        grid_resolution=grid_resolution,
        time_options=time_options,
        selected_time=selected_time,
        workspace_payload=workspace_payload,
        message=None,
    )


@app.route("/workspace/street-overlay")
def workspace_street_overlay():
    overlay = _build_street_overlay()
    return jsonify(
        {
            "available": bool(overlay.get("available", False)),
            "lat": overlay.get("lat", []),
            "lon": overlay.get("lon", []),
            "summary": overlay.get("summary", []),
            "loaded": True,
        }
    )


@app.route("/workspace/weather-overlay")
def workspace_weather_overlay():
    overlay = weather_fetcher.fetch_workspace_weather(WORKSPACE_BBOX, MAP_CENTER)
    return jsonify(overlay)


@app.route("/workspace/energy-overlay")
def workspace_energy_overlay():
    dataset_id = request.args.get("dataset_id", DEFAULT_ENERGY_DATASET_ID)
    metric = request.args.get("metric", DEFAULT_ENERGY_METRIC)
    return jsonify(_build_energy_overlay(dataset_id, metric))


@app.route("/workspace/census-overlay")
def workspace_census_overlay():
    metric = request.args.get("metric", "median_income")
    view = request.args.get("view", DEFAULT_CENSUS_VIEW)
    try:
        year_new = int(request.args.get("year_new", str(DEFAULT_CENSUS_YEAR_NEW)))
    except ValueError:
        year_new = DEFAULT_CENSUS_YEAR_NEW
    try:
        year_old = int(request.args.get("year_old", str(DEFAULT_CENSUS_YEAR_OLD)))
    except ValueError:
        year_old = DEFAULT_CENSUS_YEAR_OLD
    return jsonify(_build_census_overlay(metric, view, year_new, year_old))


@app.route("/internal/refresh-air-data", methods=["POST"])
def refresh_air_data():
    expected_secret = os.getenv(config.CRON_SECRET_ENV_VAR, "").strip()
    if not expected_secret:
        return jsonify({"ok": False, "error": f"Missing {config.CRON_SECRET_ENV_VAR} environment variable."}), 503

    if _provided_refresh_secret() != expected_secret:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    try:
        prepared = _reload_air_data(refresh=True)
        if "time" in prepared.time_series.columns:
            latest_time = pd.to_datetime(prepared.time_series["time"], errors="coerce").max()
        else:
            latest_time = pd.NaT
        latest_time_iso = latest_time.isoformat() if pd.notna(latest_time) else None
        return jsonify(
            {
                "ok": True,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "latest_observation_time": latest_time_iso,
                "data_source": prepared.source,
                "rows": int(prepared.time_series.shape[0]),
                "sensors": int(prepared.sensor_catalog.shape[0]),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/")
def overview():
    return redirect(url_for("map_workspace"))


if __name__ == "__main__":
    app.run(debug=True)
