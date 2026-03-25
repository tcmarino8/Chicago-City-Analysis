"""Flask entry point for the Chicago AQI Sensor Network experience."""
from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request
from plotly.io import to_html

try:
    import osmnx as ox
except Exception:  # pragma: no cover - optional dependency
    ox = None

from core import config, geo_utils, open_data_fetcher, weather_fetcher
import data_intake
import interpolation_models
import network_analysis


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
METHOD_OPTIONS = [("IDW", "idw"), ("Kriging", "kriging")]
WORKSPACE_METHOD_OPTIONS = [("Linear", "linear"), ("IDW", "idw"), ("Kriging", "kriging")]
STREET_MAP_PLACE_NAME = "Chicago, Illinois, USA"
ENERGY_METRIC_OPTIONS = [
    ("Energy Star Score", "energy_star_score"),
    ("Chicago Energy Rating", "chicago_energy_rating"),
]
SCHOOL_METRIC_OPTIONS = [
    ("School Type", "school_type"),
    ("Graduation Rate Mean", "graduation_rate_mean"),
]
CENSUS_METRIC_OPTIONS = [
    ("Median Household Income", "median_household_income"),
    ("White Population", "white_population"),
    ("Percent Black Pop", "black_share"),
    ("Percent Asian Pop", "asian_share"),
]
CENSUS_YEAR_OPTIONS = [2020, 2021, 2022, 2023]


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _sensor_summary() -> pd.DataFrame:
    stats = (
        PREPARED.time_series.groupby("sensor_name")
        .agg(
            observations=("aqi_value", "size"),
            avg_aqi=("aqi_value", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )
    catalog = PREPARED.sensor_catalog
    merged = catalog.merge(stats, on="sensor_name", how="left", suffixes=("", "_stat"))
    merged["observations"] = merged["observations"].fillna(0)
    merged["avg_aqi"] = merged["avg_aqi"].fillna(0.0)
    return merged


def _build_sensor_graph(summary_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in summary_df.itertuples(index=False):
        G.add_node(
            row.sensor_name,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            observations=int(row.observations),
            avg_aqi=float(row.avg_aqi),
        )
    coords_df = summary_df[["latitude", "longitude"]]
    distance_matrix = geo_utils.pairwise_distance_matrix(coords_df)
    epsilon = 1e-3
    names = summary_df["sensor_name"].to_list()
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_matrix[i, j]
            if np.isnan(dist):
                continue
            weight = 1.0 / max(dist, epsilon)
            G.add_edge(names[i], names[j], weight=weight, distance_km=float(dist))
    return G


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


def _build_uptime_fig(df: pd.DataFrame, center: dict[str, float], zoom: float) -> go.Figure:
    colors = np.where(df["is_observing"], "#34c759", "#ff3b30")
    hover = (
        "Station: "
        + df["sensor_name"].astype(str)
        + "<br>Status: "
        + np.where(df["is_observing"], "Observing", "Idle")
        + "<br>AQI: "
        + df["aqi_value"].round(2).astype(str)
    )
    fig = go.Figure(
        go.Scattermap(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            marker={"size": 12, "color": colors, "opacity": 0.85},
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox={"style": "carto-positron", "center": center, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=600,
    )
    return fig


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


def _build_network_fig(G: nx.Graph) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()
    positions = nx.spring_layout(G, weight="distance_km", seed=7, k=0.3)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "#aaa"},
        hoverinfo="none",
    )
    node_x = [positions[n][0] for n in G.nodes()]
    node_y = [positions[n][1] for n in G.nodes()]
    values = [G.nodes[n].get("aqi_value") for n in G.nodes()]
    node_color = [0.0 if value is None or (isinstance(value, float) and np.isnan(value)) else value for value in values]
    node_text = []
    for name, value in zip(G.nodes(), values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            text_value = "n/a"
        else:
            text_value = f"{value:.1f}"
        node_text.append(f"{name}<br>AQI: {text_value}")
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker={
            "size": 14,
            "color": node_color,
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {"title": "AQI"},
        },
        text=node_text,
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=600)
    return fig


def _network_metrics(G: nx.Graph) -> list[dict[str, str]]:
    df = network_analysis.compute_node_metrics(G)
    top = df.sort_values("degree", ascending=False).head(10)
    return top.to_dict("records")


def _network_summary(G: nx.Graph) -> list[str]:
    summary = network_analysis.compute_graph_summary(G)
    return [
        f"Nodes: {summary['nodes']}",
        f"Edges: {summary['edges']}",
        f"Density: {summary['density']:.3f}",
        f"Avg degree: {summary['avg_degree']:.2f}",
    ]


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


def _network_map(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df.empty:
        return go.Figure()
    sizes = summary_df["observations"].astype(float)
    marker_size = 10 + 2 * np.sqrt(sizes.clip(lower=1))
    hover = (
        "Sensor: "
        + summary_df["sensor_name"].astype(str)
        + "<br>Observations: "
        + sizes.astype(int).astype(str)
        + "<br>Avg AQI: "
        + summary_df["avg_aqi"].round(2).astype(str)
    )
    fig = go.Figure(
        go.Scattermap(
            lat=summary_df["latitude"],
            lon=summary_df["longitude"],
            mode="markers",
            marker={
                "size": marker_size,
                "color": summary_df["avg_aqi"],
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Avg AQI"},
                "opacity": 0.85,
            },
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox={"style": "carto-positron", "center": MAP_CENTER, "zoom": 9},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=650,
    )
    return fig


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
            ("Map Workspace", "map_workspace"),
            ("Overview", "overview"),
            ("Data Explorer", "data_explorer"),
            ("Network Analysis", "network_view"),
            ("Interpolation Studio", "interpolation_view"),
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
    energy_metric = request.args.get("energy_metric", "energy_star_score")
    valid_energy_metrics = {value for _, value in ENERGY_METRIC_OPTIONS}
    if energy_metric not in valid_energy_metrics:
        energy_metric = "energy_star_score"

    school_metric = request.args.get("school_metric", "school_type")
    valid_school_metrics = {value for _, value in SCHOOL_METRIC_OPTIONS}
    if school_metric not in valid_school_metrics:
        school_metric = "school_type"

    school_type = request.args.get("school_type", "all")
    school_year = request.args.get("school_year", "all")

    census_metric = request.args.get("census_metric", "black_share")
    valid_census_metrics = {value for _, value in CENSUS_METRIC_OPTIONS}
    if census_metric not in valid_census_metrics:
        census_metric = "black_share"
    census_heatmap = request.args.get("census_heatmap", "0") in {"1", "true", "True", "on"}

    try:
        census_year = int(request.args.get("census_year", str(open_data_fetcher.DEFAULT_CENSUS_YEAR)))
    except ValueError:
        census_year = int(open_data_fetcher.DEFAULT_CENSUS_YEAR)
    if census_year not in CENSUS_YEAR_OPTIONS:
        census_year = int(open_data_fetcher.DEFAULT_CENSUS_YEAR)
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
            "metric": energy_metric,
        },
        "energy_summary": ["Chicago energy metrics load on demand. Click Energy to fetch it."],
        "schools_overlay": {
            "available": False,
            "points": [],
            "loaded": False,
            "metric": school_metric,
        },
        "schools_summary": ["Chicago school profile data loads on demand. Click Schools to fetch it."],
        "census_overlay": {
            "available": False,
            "points": [],
            "loaded": False,
            "metric": census_metric,
            "census_year": census_year,
        },
        "census_heatmap_enabled": census_heatmap,
        "census_summary": ["Census tract map loads on demand. Toggle heatmap for density view."],
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
        energy_metric=energy_metric,
        energy_metric_options=ENERGY_METRIC_OPTIONS,
        school_metric=school_metric,
        school_metric_options=SCHOOL_METRIC_OPTIONS,
        school_type=school_type,
        school_year=school_year,
        census_metric=census_metric,
        census_metric_options=CENSUS_METRIC_OPTIONS,
        census_year=census_year,
        census_year_options=CENSUS_YEAR_OPTIONS,
        census_heatmap=census_heatmap,
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
    metric = request.args.get("metric", "energy_star_score")
    overlay = open_data_fetcher.fetch_energy_overlay(metric=metric)
    return jsonify(overlay)


@app.route("/workspace/schools-overlay")
def workspace_schools_overlay():
    metric = request.args.get("metric", "school_type")
    school_type = request.args.get("school_type", "all")
    school_year = request.args.get("school_year", "all")
    overlay = open_data_fetcher.fetch_schools_overlay(
        metric=metric,
        school_type=school_type,
        school_year=school_year,
    )
    return jsonify(overlay)


@app.route("/workspace/census-overlay")
def workspace_census_overlay():
    metric = request.args.get("metric", "black_share")
    try:
        census_year = int(request.args.get("year", str(open_data_fetcher.DEFAULT_CENSUS_YEAR)))
    except ValueError:
        census_year = int(open_data_fetcher.DEFAULT_CENSUS_YEAR)
    overlay = open_data_fetcher.fetch_census_overlay(metric=metric, census_year=census_year)
    return jsonify(overlay)


@app.route("/")
def overview():
    goals = [
        "Understand Chicago's air-quality coverage using open Socrata data.",
        "Explain modeling choices spanning interpolation, network dynamics, and (future) STGNNs.",
        "Share interactive visuals that can be explored without a notebook runtime.",
    ]
    highlights = [
        "Paginated Socrata ingestion eliminates the 5k-row cap.",
        "Modules for intake, network analysis, and interpolation keep logic reusable.",
        "Flask site stitches narrative, stats, and visualizations into one shareable hub.",
    ]
    return render_template("overview.html", goals=goals, highlights=highlights)


@app.route("/data-explorer")
def data_explorer():
    bin_code = request.args.get("bin", config.DEFAULT_BIN)
    coverage_str = request.args.get("coverage", str(config.DEFAULT_COVERAGE_THRESHOLD))
    coverage_threshold = _safe_float(coverage_str, config.DEFAULT_COVERAGE_THRESHOLD)
    center_lat = _safe_float(request.args.get("center_lat"), MAP_CENTER["lat"])
    center_lon = _safe_float(request.args.get("center_lon"), MAP_CENTER["lon"])
    zoom = _safe_float(request.args.get("zoom"), 9.0)
    map_center = {"lat": center_lat, "lon": center_lon}
    time_options = _time_options(bin_code, coverage_threshold)
    if not time_options:
        aggregated = get_aggregated(bin_code)
        if aggregated.empty:
            message = "No aggregated data available for the selected bin."
        else:
            message = "No time bins meet the selected coverage requirement."
        return render_template(
            "data_explorer.html",
            bin_code=bin_code,
            bin_options=config.BIN_OPTIONS,
            time_options=[],
            selected_time=None,
            coverage_threshold=coverage_threshold,
            coverage_options=config.COVERAGE_THRESHOLDS,
            map_center=map_center,
            zoom=zoom,
            plot_html=None,
            stats_list=[],
            message=message,
        )
    selected_time = request.args.get("time")
    if not selected_time or all(opt["value"] != selected_time for opt in time_options):
        selected_time = time_options[0]["value"]
    aggregated = get_aggregated(bin_code)
    time_value = pd.to_datetime(selected_time)
    merged, stats = data_intake.compute_uptime(aggregated, PREPARED.sensor_catalog, time_value, coverage_threshold)
    fig = _build_uptime_fig(merged, center=map_center, zoom=zoom)
    plot_html = to_html(fig, full_html=False, include_plotlyjs="cdn")
    stats_list = _uptime_stats(stats)
    return render_template(
        "data_explorer.html",
        bin_code=bin_code,
        bin_options=config.BIN_OPTIONS,
        time_options=time_options,
        selected_time=selected_time,
        coverage_threshold=coverage_threshold,
        coverage_options=config.COVERAGE_THRESHOLDS,
        map_center=map_center,
        zoom=zoom,
        plot_html=plot_html,
        stats_list=stats_list,
        message=None,
    )


@app.route("/network")
def network_view():
    summary_df = _sensor_summary()
    if summary_df.empty:
        return render_template(
            "network.html",
            plot_html=None,
            summary_list=[],
            metrics=[],
            message="No sensor data available.",
        )
    G = _build_sensor_graph(summary_df)
    map_fig = _network_map(summary_df)
    plot_html = to_html(map_fig, full_html=False, include_plotlyjs="cdn")
    summary_list = _network_summary(G)
    top_nodes = (
        summary_df.sort_values("observations", ascending=False)
        .head(5)
        .loc[:, ["sensor_name", "latitude", "longitude", "observations", "avg_aqi"]]
        .assign(
            observations=lambda df: df["observations"].astype(int),
            avg_aqi=lambda df: df["avg_aqi"].astype(float),
        )
        .to_dict("records")
    )
    return render_template(
        "network.html",
        plot_html=plot_html,
        summary_list=summary_list,
        metrics=top_nodes,
        message=None,
    )


@app.route("/interpolation")
def interpolation_view():
    bin_code = request.args.get("bin", config.DEFAULT_BIN)
    coverage_str = request.args.get("coverage", str(config.DEFAULT_COVERAGE_THRESHOLD))
    coverage_threshold = _safe_float(coverage_str, config.DEFAULT_COVERAGE_THRESHOLD)
    coverage_threshold = min(max(coverage_threshold, 0.0), 1.0)
    method = request.args.get("method", "idw")
    grid_res_str = request.args.get("grid", str(config.DEFAULT_GRID_RESOLUTION))
    try:
        grid_resolution = int(grid_res_str)
    except ValueError:
        grid_resolution = config.DEFAULT_GRID_RESOLUTION
    time_options = _time_options(bin_code, coverage_threshold)
    uptime_message = None
    interpolation_message = None
    uptime_plot_html = None
    interpolation_plot_html = None
    uptime_stats_list: list[str] = []
    interpolation_stats: list[str] = []
    if not time_options:
        aggregated = get_aggregated(bin_code)
        if aggregated.empty:
            uptime_message = "No aggregated data available for the selected bin."
        else:
            uptime_message = "No time bins meet the selected coverage requirement."
        return render_template(
            "interpolation.html",
            bin_code=bin_code,
            bin_options=config.BIN_OPTIONS,
            coverage_threshold=coverage_threshold,
            coverage_options=config.COVERAGE_THRESHOLDS,
            time_options=[],
            selected_time=None,
            method=method,
            method_options=METHOD_OPTIONS,
            grid_resolution=grid_resolution,
            uptime_plot_html=uptime_plot_html,
            uptime_stats=uptime_stats_list,
            interpolation_plot_html=interpolation_plot_html,
            interpolation_stats=interpolation_stats,
            uptime_message=uptime_message,
            interpolation_message=interpolation_message,
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
    uptime_fig = _build_uptime_fig(merged, center=MAP_CENTER, zoom=9.0)
    uptime_plot_html = to_html(uptime_fig, full_html=False, include_plotlyjs="cdn")
    uptime_stats_list = _uptime_stats(uptime_stats)
    df_slice = data_intake.filter_by_time(aggregated, time_value)
    result = interpolation_models.interpolate_time_slice(df_slice, method=method, grid_resolution=grid_resolution)
    if result is None:
        interpolation_message = "Need at least three observing stations to interpolate."
    else:
        contour = go.Contour(
            x=result.grid_lon[0],
            y=result.grid_lat[:, 0],
            z=result.grid_values,
            colorscale="Turbo",
            contours_coloring="heatmap",
            colorbar={"title": "AQI"},
            showscale=True,
        )
        scatter = go.Scatter(
            x=result.source_points["longitude"],
            y=result.source_points["latitude"],
            mode="markers",
            marker={"color": "black", "size": 6},
            text=result.source_points["sensor_name"],
            name="Sensors",
        )
        fig = go.Figure([contour, scatter])
        fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude", height=650)
        interpolation_plot_html = to_html(fig, full_html=False, include_plotlyjs="cdn")
        interpolation_stats = _interpolation_stats(result)
    return render_template(
        "interpolation.html",
        bin_code=bin_code,
        bin_options=config.BIN_OPTIONS,
        coverage_threshold=coverage_threshold,
        coverage_options=config.COVERAGE_THRESHOLDS,
        time_options=time_options,
        selected_time=selected_time,
        method=method,
        method_options=METHOD_OPTIONS,
        grid_resolution=grid_resolution,
        uptime_plot_html=uptime_plot_html,
        uptime_stats=uptime_stats_list,
        interpolation_plot_html=interpolation_plot_html,
        interpolation_stats=interpolation_stats,
        uptime_message=uptime_message,
        interpolation_message=interpolation_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
