"""Flask entry point for the Chicago AQI Sensor Network experience."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

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
PREPARED = data_intake.get_prepared_data()
MAP_CENTER = {
    "lat": PREPARED.sensor_catalog["latitude"].mean(),
    "lon": PREPARED.sensor_catalog["longitude"].mean(),
}
WORKSPACE_BBOX = geo_utils.BoundingBox.from_points(
    PREPARED.sensor_catalog["latitude"],
    PREPARED.sensor_catalog["longitude"],
    padding=0.1,
)
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


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


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

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
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


@lru_cache(maxsize=len(config.BIN_OPTIONS))
def get_aggregated(bin_code: str) -> pd.DataFrame:
    return data_intake.aggregate_time_series(PREPARED.time_series, bin_code)


def _time_options(bin_code: str, min_coverage: float = 0.0) -> list[dict[str, str]]:
    df = get_aggregated(bin_code)
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


def _interpolation_stats(result: interpolation_models.InterpolationResult) -> list[str]:
    return [
        f"Method: {result.method_used.upper()}",
        f"Min AQI: {result.grid_values.min():.1f}",
        f"Mean AQI: {result.grid_values.mean():.1f}",
        f"Max AQI: {result.grid_values.max():.1f}",
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
    bin_code = request.args.get("bin", config.DEFAULT_BIN)
    coverage_str = request.args.get("coverage", str(config.DEFAULT_COVERAGE_THRESHOLD))
    coverage_threshold = min(max(_safe_float(coverage_str, config.DEFAULT_COVERAGE_THRESHOLD), 0.0), 1.0)
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
    grid_resolution = int(np.clip(grid_resolution, 30, 200))

    time_options = _time_options(bin_code, coverage_threshold)
    if not time_options:
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

    aggregated = get_aggregated(bin_code)
    time_value = pd.to_datetime(selected_time)
    merged, uptime_stats = data_intake.compute_uptime(
        aggregated,
        PREPARED.sensor_catalog,
        time_value,
        coverage_threshold,
    )
    sensor_max_aqi = aggregated.groupby("sensor_name")["aqi_value"].max()
    merged["max_aqi"] = merged["sensor_name"].map(sensor_max_aqi)

    distance_source = merged.copy() if show_all_points else merged[merged["is_observing"]].copy()
    distance_source = distance_source.dropna(subset=["latitude", "longitude"])
    condensed_dist = geo_utils.haversine_matrix(distance_source)
    distance_options_km: list[float] = []
    selected_distance_km = config.ADJACENCY_DISTANCE_THRESHOLD_KM
    if condensed_dist.size > 0:
        positive_dist = condensed_dist[condensed_dist > 0]
        if positive_dist.size == 0:
            positive_dist = np.array([1.0])

        dist_min = float(np.min(positive_dist))
        dist_max = float(np.max(positive_dist))
        if np.isclose(dist_min, dist_max):
            distance_options_km = [round(dist_min, 2)]
        else:
            # Log-like spacing, anchored to avoid near-zero thresholds.
            # For a ~20 km max this yields options close to: 2, 5, 11, 20.
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
                "max_aqi": None if pd.isna(row.max_aqi) else float(row.max_aqi),
                "is_observing": is_observing,
                "observation_count": obs_count,
                "opacity": 0.22 if (show_all_points and not is_observing) else 0.95,
            }
        )

    network_overlay = _build_network_overlay(
        merged,
        show_all_points=show_all_points,
        threshold_km=selected_distance_km,
    )
    df_slice = data_intake.filter_by_time(aggregated, time_value)

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
        interpolation_stats = _interpolation_stats(interpolation_result)

    workspace_payload = {
        "map_center": MAP_CENTER,
        "time": time_value.strftime("%Y-%m-%d %H:%M"),
        "active_mode": request.args.get("mode", "explorer"),
        "show_all_points": show_all_points,
        "selected_distance_km": selected_distance_km,
        "distance_options_km": distance_options_km,
        "uptime_points": uptime_points,
        "uptime_stats": _uptime_stats(uptime_stats),
        "network_overlay": network_overlay,
        "network_summary": _network_overlay_summary(network_overlay, selected_distance_km),
        "interpolation_points": interpolation_points,
        "interpolation_stats": interpolation_stats,
        "interpolation_method_used": interpolation_result.method_used if interpolation_result else method,
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
        "max_aqi_range": {
            "min": float(np.nanmin(sensor_max_aqi.values)) if len(sensor_max_aqi) else 0.0,
            "max": float(np.nanmax(sensor_max_aqi.values)) if len(sensor_max_aqi) else 1.0,
        },
    }

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


@app.route("/")
def overview():
    return redirect(url_for("map_workspace"))


if __name__ == "__main__":
    app.run(debug=True)
