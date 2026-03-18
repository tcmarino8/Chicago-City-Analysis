"""Dash page: exploratory stats + uptime map."""
from __future__ import annotations

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from core import config
import data_intake


dash.register_page(__name__, path="/explorer", name="Data Explorer")

PREPARED = data_intake.get_prepared_data()
MAP_CENTER = {
    "lat": PREPARED.sensor_catalog["latitude"].mean(),
    "lon": PREPARED.sensor_catalog["longitude"].mean(),
}


@lru_cache(maxsize=len(config.BIN_OPTIONS))
def get_aggregated(bin_code: str) -> pd.DataFrame:
    return data_intake.aggregate_time_series(PREPARED.time_series, bin_code)


def _time_options(bin_code: str) -> list[dict]:
    df = get_aggregated(bin_code)
    return [
        {"label": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M"), "value": pd.to_datetime(ts).isoformat()}
        for ts in df["time"].drop_duplicates()
    ]


def _build_uptime_fig(df: pd.DataFrame) -> go.Figure:
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
        go.Scattermapbox(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            marker={"size": 12, "color": colors, "opacity": 0.85},
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox={"style": "carto-positron", "center": MAP_CENTER, "zoom": 12},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=600,
    )
    return fig


def _stats_panel(stats: dict) -> html.Div:
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
    return html.Ul([html.Li(item) for item in entries])


layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id="bin-dropdown",
                    options=[{"label": label, "value": label} for label in config.BIN_OPTIONS],
                    value=config.DEFAULT_BIN,
                    clearable=False,
                ),
                dcc.Dropdown(id="time-dropdown", options=[], placeholder="Select time bin"),
                dcc.Dropdown(
                    id="coverage-dropdown",
                    options=[{"label": f">= {int(val*100)}%", "value": val} for val in config.COVERAGE_THRESHOLDS],
                    value=config.DEFAULT_COVERAGE_THRESHOLD,
                    clearable=False,
                ),
            ],
            className="control-row",
        ),
        dcc.Graph(id="uptime-map"),
        html.Div(id="stats-panel"),
    ],
    className="page explorer-page",
)


@callback(Output("time-dropdown", "options"), Output("time-dropdown", "value"), Input("bin-dropdown", "value"))
def update_time_dropdown(bin_code: str):
    options = _time_options(bin_code)
    value = options[0]["value"] if options else None
    return options, value


@callback(
    Output("uptime-map", "figure"),
    Output("stats-panel", "children"),
    Input("bin-dropdown", "value"),
    Input("time-dropdown", "value"),
    Input("coverage-dropdown", "value"),
)
def update_map(bin_code: str, iso_time: str | None, coverage_threshold: float):
    if not iso_time:
        return go.Figure(), html.Div("No data available")
    aggregated = get_aggregated(bin_code)
    time_value = pd.to_datetime(iso_time)
    merged, stats = data_intake.compute_uptime(aggregated, PREPARED.sensor_catalog, time_value, coverage_threshold)
    fig = _build_uptime_fig(merged)
    return fig, _stats_panel(stats)
