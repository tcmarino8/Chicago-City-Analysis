"""Dash page: network analysis views."""
from __future__ import annotations

from functools import lru_cache

import dash
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from core import config
import data_intake
import network_analysis


dash.register_page(__name__, path="/network", name="Network Analysis")

PREPARED = data_intake.get_prepared_data()


@lru_cache(maxsize=len(config.BIN_OPTIONS))
def get_aggregated(bin_code: str) -> pd.DataFrame:
    return data_intake.aggregate_time_series(PREPARED.time_series, bin_code)


def _time_options(bin_code: str) -> list[dict]:
    return [
        {"label": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M"), "value": pd.to_datetime(ts).isoformat()}
        for ts in get_aggregated(bin_code)["time"].drop_duplicates()
    ]


def _graph_figure(G: nx.Graph) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()
    positions = nx.spring_layout(G, weight="distance_km", seed=7, k=0.3)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
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
    node_color = [0.0 if value is None else value for value in values]
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


def _stats_panel(G: nx.Graph) -> html.Div:
    summary = network_analysis.compute_graph_summary(G)
    metrics = network_analysis.compute_node_metrics(G)
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Sensor"), html.Th("Degree"), html.Th("Betweenness"), html.Th("Closeness")]))
        ]
        + [
            html.Tr(
                [
                    html.Td(row.sensor_name),
                    html.Td(f"{row.degree:.0f}"),
                    html.Td(f"{row.betweenness:.3f}"),
                    html.Td(f"{row.closeness:.3f}"),
                ]
            )
            for row in metrics.sort_values("degree", ascending=False).head(10).itertuples()
        ]
    )
    summary_list = html.Ul(
        [
            html.Li(f"Nodes: {summary['nodes']}"),
            html.Li(f"Edges: {summary['edges']}"),
            html.Li(f"Density: {summary['density']:.3f}"),
            html.Li(f"Avg degree: {summary['avg_degree']:.2f}"),
        ]
    )
    return html.Div([summary_list, table], className="network-stats")


layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id="network-bin-dropdown",
                    options=[{"label": label, "value": label} for label in config.BIN_OPTIONS],
                    value=config.DEFAULT_BIN,
                    clearable=False,
                ),
                dcc.Dropdown(id="network-time-dropdown", options=[], placeholder="Select time slice"),
                dcc.Slider(
                    id="threshold-slider",
                    min=2,
                    max=20,
                    step=1,
                    value=config.ADJACENCY_DISTANCE_THRESHOLD_KM,
                    marks={i: f"{i} km" for i in range(2, 21, 3)},
                ),
            ],
            className="control-row",
        ),
        dcc.Graph(id="network-graph"),
        html.Div(id="network-stats-panel"),
    ],
    className="page network-page",
)


@callback(
    Output("network-time-dropdown", "options"),
    Output("network-time-dropdown", "value"),
    Input("network-bin-dropdown", "value"),
)
def update_network_time(bin_code: str):
    options = _time_options(bin_code)
    value = options[0]["value"] if options else None
    return options, value


@callback(
    Output("network-graph", "figure"),
    Output("network-stats-panel", "children"),
    Input("network-bin-dropdown", "value"),
    Input("network-time-dropdown", "value"),
    Input("threshold-slider", "value"),
)
def render_network(bin_code: str, iso_time: str | None, threshold_km: float):
    if not iso_time:
        return go.Figure(), html.Div("No graph available")
    aggregated = get_aggregated(bin_code)
    df_slice = data_intake.filter_by_time(aggregated, pd.to_datetime(iso_time))
    adjacency, distances = network_analysis.build_adjacency(PREPARED.sensor_catalog, threshold_km)
    G = network_analysis.graph_from_catalog(PREPARED.sensor_catalog, adjacency, distances)
    network_analysis.annotate_time_slice(G, df_slice)
    return _graph_figure(G), _stats_panel(G)
