"""Plotly Dash app for visualizing Chicago AQI sensor uptime by aggregation window.

The app recreates the notebook widget that shows which sensors reported data in a
selected time bin and overlays summary statistics. Run the script locally to get a
browser-based preview before deploying anywhere else.

Usage:
    python sensor_uptime_app.py --limit 8000 --cache-csv chicago_aqi_cache.csv

Environment variables:
    SOCRATA_APP_TOKEN (optional): reuse an app token for higher rate limits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from sodapy import Socrata

DATASET_ID = "xfya-dxtq"
DEFAULT_AQI_COLUMN = "pm2_5concmassindividual_value"
DEFAULT_LIMIT = 5000
BIN_OPTIONS = [
    {"label": "15 minutes", "value": "15min"},
    {"label": "1 hour", "value": "1h"},
    {"label": "6 hours", "value": "6h"},
    {"label": "12 hours", "value": "12h"},
    {"label": "1 day", "value": "1d"},
    {"label": "1 week", "value": "7d"},
]
BIN_LABEL_LOOKUP = {opt["value"]: opt["label"] for opt in BIN_OPTIONS}
COVERAGE_OPTIONS = [
    {"label": "0%+ sensors", "value": 0.0},
    {"label": "25%+ sensors", "value": 0.25},
    {"label": "50%+ sensors", "value": 0.50},
    {"label": "60%+ sensors", "value": 0.60},
    {"label": "75%+ sensors", "value": 0.75},
]
SENSOR_CANDIDATES = ["sensor_name", "sensor", "sensor_id", "name", "id", "site_id"]

DF_TS: pd.DataFrame
SENSOR_CATALOG: pd.DataFrame
AGGREGATE_CACHE: Dict[str, pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chicago AQI uptime dashboard")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Rows to request from Socrata")
    parser.add_argument("--aqi-column", type=str, default=DEFAULT_AQI_COLUMN,
                        help="Column to use as the air quality metric")
    parser.add_argument("--cache-csv", type=str, default=None,
                        help="Optional CSV cache path. Loaded if present, written after API fetch.")
    parser.add_argument("--app-token", type=str, default=os.getenv("SOCRATA_APP_TOKEN"),
                        help="Socrata app token (defaults to SOCRATA_APP_TOKEN env var).")
    return parser.parse_args()


def load_dataframe(limit: int, cache_csv: Optional[str], app_token: Optional[str]) -> pd.DataFrame:
    if cache_csv:
        cache_path = Path(cache_csv)
        if cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            return pd.read_csv(cache_path)
    print(f"Fetching up to {limit} rows from Socrata dataset {DATASET_ID}")
    client = Socrata("data.cityofchicago.org", app_token)
    records = client.get(DATASET_ID, limit=limit)
    df = pd.DataFrame.from_records(records)
    if cache_csv:
        cache_path = Path(cache_csv)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached API response to {cache_path}")
    return df


def extract_coordinates(cell):
    if pd.isna(cell):
        return None, None
    if isinstance(cell, dict):
        if {"latitude", "longitude"}.issubset(cell):
            return cell["latitude"], cell["longitude"]
        if "coordinates" in cell and len(cell["coordinates"]) >= 2:
            lon, lat = cell["coordinates"][:2]
            return lat, lon
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            return extract_coordinates(parsed)
        except json.JSONDecodeError:
            return None, None
    return None, None


def ensure_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    if {"latitude", "longitude"}.issubset(df.columns):
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        return df
    if "location" in df.columns:
        coords = df["location"].apply(extract_coordinates)
        df["latitude"] = coords.apply(lambda item: item[0])
        df["longitude"] = coords.apply(lambda item: item[1])
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        return df
    raise ValueError("Dataset must include latitude/longitude or a 'location' column with coordinates.")


def identify_sensor_column(df: pd.DataFrame) -> str:
    for candidate in SENSOR_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError("Unable to locate a sensor identifier column.")


def prepare_dataframe(df: pd.DataFrame, aqi_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensor_col = identify_sensor_column(df)
    if sensor_col != "sensor_name":
        df = df.rename(columns={sensor_col: "sensor_name"})
    df = ensure_coordinates(df)
    if "time" not in df.columns:
        raise ValueError("Dataset does not contain a 'time' column.")
    if aqi_column not in df.columns:
        raise ValueError(f"Column '{aqi_column}' not found. Pass --aqi-column if the metric has a different name.")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[aqi_column] = pd.to_numeric(df[aqi_column], errors="coerce")
    df = df.dropna(subset=["sensor_name", "latitude", "longitude", "time", aqi_column])
    df = df.sort_values("time").reset_index(drop=True)
    df_sorted = df.sort_values("time")
    latest_locations = (
        df_sorted[["sensor_name", "latitude", "longitude", "time"]]
        .drop_duplicates(subset="sensor_name", keep="last")
    )
    observation_counts = (
        df_sorted.groupby("sensor_name")
        .size()
        .rename("total_obs")
        .reset_index()
    )
    sensor_catalog = (
        latest_locations
        .merge(observation_counts, on="sensor_name")
        [["sensor_name", "latitude", "longitude", "total_obs"]]
        .reset_index(drop=True)
    )
    df_ts = df_sorted[["sensor_name", "latitude", "longitude", "time", aqi_column]].copy()
    df_ts = df_ts.rename(columns={aqi_column: "aqi_value"})
    return df_ts, sensor_catalog


def aggregate_for_bin(bin_code: str) -> pd.DataFrame:
    if bin_code in AGGREGATE_CACHE:
        return AGGREGATE_CACHE[bin_code]
    df_tmp = DF_TS.copy()
    df_tmp["time_bin"] = df_tmp["time"].dt.floor(bin_code)
    agg = (
        df_tmp.groupby(["sensor_name", "time_bin", "latitude", "longitude"]) ["aqi_value"]
        .agg(aqi_value="mean", reading_count="size")
        .reset_index()
        .rename(columns={"time_bin": "time"})
        .sort_values("time")
    )
    AGGREGATE_CACHE[bin_code] = agg
    return agg


def haversine_distances(df_points: pd.DataFrame) -> np.ndarray:
    if len(df_points) < 2:
        return np.array([])
    coords = np.radians(df_points[["latitude", "longitude"]].to_numpy())
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * (np.sin(dlon / 2) ** 2)
    distances = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return distances[np.triu_indices(len(df_points), k=1)]


def distance_statistics(df_points: pd.DataFrame) -> Dict[str, float]:
    dists = haversine_distances(df_points)
    if dists.size == 0:
        return {"range": 0.0, "mean": 0.0, "min": 0.0}
    return {"range": float(dists.max()), "mean": float(dists.mean()), "min": float(dists.min())}


def build_status_frame(df_time: pd.DataFrame) -> pd.DataFrame:
    status = SENSOR_CATALOG.copy()
    active = set(df_time["sensor_name"])
    status["status"] = np.where(status["sensor_name"].isin(active), "Observing", "Not observing")
    status = status.merge(
        df_time[["sensor_name", "aqi_value", "reading_count"]],
        on="sensor_name",
        how="left",
    )
    status["bin_readings"] = status.pop("reading_count").fillna(0).astype(int)
    status["total_obs"] = status["total_obs"].fillna(0).astype(int)
    return status


def format_stats(bin_code: str, selected_time: pd.Timestamp, df_time: pd.DataFrame) -> Dict[str, str]:
    status_counts = {
        "total": len(SENSOR_CATALOG),
        "observing": df_time["sensor_name"].nunique(),
    }
    status_counts["not_observing"] = status_counts["total"] - status_counts["observing"]
    coverage_pct = (
        (status_counts["observing"] / status_counts["total"]) * 100
        if status_counts["total"]
        else 0.0
    )
    mean_val = df_time["aqi_value"].mean() if status_counts["observing"] else np.nan
    std_val = df_time["aqi_value"].std(ddof=0) if status_counts["observing"] else np.nan
    dist_stats = distance_statistics(df_time)
    lines = [
        f"Bin: {BIN_LABEL_LOOKUP.get(bin_code, bin_code)} ({bin_code})",
        f"Time: {selected_time}",
        f"Observing: {status_counts['observing']} / {status_counts['total']}",
        f"Not observing: {status_counts['not_observing']}",
        f"Coverage: {coverage_pct:.1f}%",
        f"Mean value: {mean_val:.2f}" if not np.isnan(mean_val) else "Mean value: N/A",
        f"Std: {std_val:.2f}" if not np.isnan(std_val) else "Std: N/A",
        f"Spatial range (km): {dist_stats['range']:.1f}",
        f"Mean dist (km): {dist_stats['mean']:.1f}",
        f"Min dist (km): {dist_stats['min']:.1f}",
    ]
    html_lines = [
        f"<strong>{BIN_LABEL_LOOKUP.get(bin_code, bin_code)}</strong> ({bin_code})",
        selected_time.strftime("%Y-%m-%d %H:%M"),
        f"Observing: {status_counts['observing']} / {status_counts['total']}",
        f"Not observing: {status_counts['not_observing']}",
        f"Coverage: {coverage_pct:.1f}%",
        (f"Mean value: {mean_val:.2f}" if not np.isnan(mean_val) else "Mean value: N/A"),
        (f"Std: {std_val:.2f}" if not np.isnan(std_val) else "Std: N/A"),
        f"Spatial range: {dist_stats['range']:.1f} km",
        f"Mean distance: {dist_stats['mean']:.1f} km",
        f"Min distance: {dist_stats['min']:.1f} km",
    ]
    return {"annotation": "<br>".join(lines), "panel": html_lines}


def build_figure(status_df: pd.DataFrame, stats_text: str) -> px.scatter_map:
    fig = px.scatter_map(
        status_df,
        lat="latitude",
        lon="longitude",
        color="status",
        color_discrete_map={"Observing": "green", "Not observing": "red"},
        hover_name="sensor_name",
        hover_data={
            "aqi_value": ':.2f',
            "bin_readings": ':.0f',
            "total_obs": ':.0f',
            "status": False,
            "latitude": ':.4f',
            "longitude": ':.4f',
        },
        zoom=9,
        height=650,
    )
    fig.update_traces(marker=dict(size=np.where(status_df["status"] == "Observing", 14, 9)))
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        title="Chicago sensor uptime by aggregation window",
    )
    fig.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6,
        align="left",
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
    )
    return fig


def make_empty_figure(message: str):
    fig = px.scatter_map(lat=[], lon=[])
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=10, r=10, t=40, b=10),
        title=message,
    )
    return fig


def time_dropdown_options(bin_code: str, min_share: float) -> List[Dict[str, str]]:
    agg = aggregate_for_bin(bin_code)
    if agg.empty:
        return []
    total_sensors = len(SENSOR_CATALOG) or 1
    coverage = agg.groupby("time")["sensor_name"].nunique() / total_sensors
    coverage = coverage[coverage >= min_share]
    if coverage.empty:
        return []
    times = coverage.index.sort_values()
    return [
        {
            "label": f"{pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')} ({coverage.loc[ts]*100:.0f}% sensors)",
            "value": pd.to_datetime(ts).isoformat(),
        }
        for ts in times
    ]


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("time-dropdown", "options"),
        Output("time-dropdown", "value"),
        Input("bin-dropdown", "value"),
        Input("coverage-dropdown", "value"),
    )
    def update_time_options(bin_code: Optional[str], min_share: Optional[float]):  # type: ignore
        if bin_code is None:
            return [], None
        min_share = min_share or 0.0
        options = time_dropdown_options(bin_code, min_share)
        if not options:
            return [], None
        return options, options[0]["value"]

    @app.callback(
        Output("uptime-map", "figure"),
        Output("stats-panel", "children"),
        Input("bin-dropdown", "value"),
        Input("time-dropdown", "value"),
    )
    def update_map(bin_code: Optional[str], time_value: Optional[str]):  # type: ignore
        if not bin_code:
            return make_empty_figure("Select a time bin"), "Select a bin to view stats."
        agg = aggregate_for_bin(bin_code)
        if agg.empty:
            return make_empty_figure("No data for this bin"), "No sensor data after preprocessing."
        if not time_value:
            return make_empty_figure("Select a timestamp"), "Pick a timestamp from the dropdown."
        selected_time = pd.to_datetime(time_value)
        df_time = agg[agg["time"] == selected_time]
        status_df = build_status_frame(df_time)
        stats_text = format_stats(bin_code, selected_time, df_time)
        fig = build_figure(status_df, stats_text["annotation"])
        panel_children = [html.Div(line) for line in stats_text["panel"]]
        return fig, panel_children


def build_app() -> Dash:
    app = Dash(__name__)
    app.title = "Chicago AQI Uptime"
    app.layout = html.Div(
        [
            html.H2("Chicago AQI Sensor Uptime"),
            html.P(
                "Visualize when air-quality sensors report values. Select a bin to aggregate "
                "readings, then pick a timestamp to see observing (green) vs offline (red) stations."
            ),
            html.Div(
                [
                    html.Div([
                        html.Label("Aggregation window"),
                        dcc.Dropdown(
                            id="bin-dropdown",
                            options=BIN_OPTIONS,
                            value="12h",
                            clearable=False,
                        ),
                    ], style={"flex": "1", "marginRight": "16px"}),
                    html.Div([
                        html.Label("Timestamp"),
                        dcc.Dropdown(id="time-dropdown", options=[], placeholder="Select a time"),
                    ], style={"flex": "2", "marginRight": "16px"}),
                    html.Div([
                        html.Label("Minimum coverage"),
                        dcc.Dropdown(
                            id="coverage-dropdown",
                            options=COVERAGE_OPTIONS,
                            value=0.0,
                            clearable=False,
                        ),
                    ], style={"flex": "1"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            dcc.Graph(id="uptime-map"),
            html.Div(id="stats-panel", className="stats-panel", style={"marginTop": "12px"}),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    )
    register_callbacks(app)
    return app


def main():
    global DF_TS, SENSOR_CATALOG
    args = parse_args()
    df_raw = load_dataframe(args.limit, args.cache_csv, args.app_token)
    DF_TS, SENSOR_CATALOG = prepare_dataframe(df_raw, args.aqi_column)
    if DF_TS.empty:
        raise SystemExit("No usable records after preprocessing. Check the AQI column and source data.")
    print(f"Prepared {len(DF_TS)} records across {DF_TS['sensor_name'].nunique()} sensors.")
    app = build_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
