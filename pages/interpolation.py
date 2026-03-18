"""Dash page: spatial interpolation playground."""
from __future__ import annotations

from functools import lru_cache

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from core import config
import data_intake
import interpolation_models


dash.register_page(__name__, path="/interpolation", name="Interpolation Studio")

PREPARED = data_intake.get_prepared_data()
METHOD_OPTIONS = [("IDW", "idw"), ("Kriging", "kriging")]


@lru_cache(maxsize=len(config.BIN_OPTIONS))
def get_aggregated(bin_code: str) -> pd.DataFrame:
    return data_intake.aggregate_time_series(PREPARED.time_series, bin_code)


def _time_options(bin_code: str) -> list[dict]:
    return [
        {"label": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M"), "value": pd.to_datetime(ts).isoformat()}
        for ts in get_aggregated(bin_code)["time"].drop_duplicates()
    ]


def _build_figure(result: interpolation_models.InterpolationResult | None) -> go.Figure:
    if result is None:
        return go.Figure()
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
    fig.update_layout(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=650,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )
    return fig


def _stats_panel(result: interpolation_models.InterpolationResult | None) -> html.Div:
    if result is None:
        return html.Div("Need at least 3 observing stations for interpolation.")
    stats = {
        "Method": result.method_used.upper(),
        "Min AQI": f"{result.grid_values.min():.1f}",
        "Mean AQI": f"{result.grid_values.mean():.1f}",
        "Max AQI": f"{result.grid_values.max():.1f}",
    }
    return html.Ul([html.Li(f"{k}: {v}") for k, v in stats.items()])


layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id="interp-bin-dropdown",
                    options=[{"label": label, "value": label} for label in config.BIN_OPTIONS],
                    value=config.DEFAULT_BIN,
                    clearable=False,
                ),
                dcc.Dropdown(id="interp-time-dropdown", options=[], placeholder="Select time slice"),
                dcc.Dropdown(
                    id="interp-method-dropdown",
                    options=[{"label": label, "value": value} for label, value in METHOD_OPTIONS],
                    value="idw",
                    clearable=False,
                ),
                dcc.Slider(
                    id="grid-resolution-slider",
                    min=40,
                    max=120,
                    step=10,
                    value=config.DEFAULT_GRID_RESOLUTION,
                    marks={val: str(val) for val in range(40, 121, 20)},
                ),
            ],
            className="control-row",
        ),
        dcc.Graph(id="interpolation-figure"),
        html.Div(id="interpolation-stats"),
    ],
    className="page interpolation-page",
)


@callback(
    Output("interp-time-dropdown", "options"),
    Output("interp-time-dropdown", "value"),
    Input("interp-bin-dropdown", "value"),
)
def update_interpolation_time(bin_code: str):
    options = _time_options(bin_code)
    value = options[0]["value"] if options else None
    return options, value


@callback(
    Output("interpolation-figure", "figure"),
    Output("interpolation-stats", "children"),
    Input("interp-bin-dropdown", "value"),
    Input("interp-time-dropdown", "value"),
    Input("interp-method-dropdown", "value"),
    Input("grid-resolution-slider", "value"),
)
def update_interpolation(bin_code: str, iso_time: str | None, method: str, grid_resolution: int):
    if not iso_time:
        return go.Figure(), html.Div("No data available")
    aggregated = get_aggregated(bin_code)
    df_slice = data_intake.filter_by_time(aggregated, pd.to_datetime(iso_time))
    result = interpolation_models.interpolate_time_slice(df_slice, method=method, grid_resolution=grid_resolution)
    return _build_figure(result), _stats_panel(result)
