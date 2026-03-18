"""Network analysis utilities built on top of NetworkX."""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from core import config, geo_utils


def build_adjacency(sensor_catalog: pd.DataFrame, threshold_km: float = config.ADJACENCY_DISTANCE_THRESHOLD_KM) -> tuple[np.ndarray, np.ndarray]:
    distances = geo_utils.pairwise_distance_matrix(sensor_catalog)
    adjacency = (distances <= threshold_km).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency, distances


def graph_from_catalog(sensor_catalog: pd.DataFrame, adjacency: np.ndarray, distances: np.ndarray | None = None) -> nx.Graph:
    sensors = sensor_catalog.reset_index(drop=True)
    G = nx.Graph()
    for row in sensors.itertuples(index=True):
        G.add_node(
            row.sensor_name,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            index=row.Index,
        )
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                attrs = {}
                if distances is not None:
                    attrs["distance_km"] = float(distances[i, j])
                    attrs["weight"] = float(max(distances[i, j], 1e-6))
                G.add_edge(sensors.loc[i, "sensor_name"], sensors.loc[j, "sensor_name"], **attrs)
    return G


def annotate_time_slice(G: nx.Graph, df_time_slice: pd.DataFrame) -> nx.Graph:
    for row in df_time_slice.itertuples():
        if row.sensor_name in G.nodes:
            G.nodes[row.sensor_name]["aqi_value"] = float(row.aqi_value)
            G.nodes[row.sensor_name]["reading_count"] = int(row.reading_count)
    return G


def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight") if G.number_of_edges() else {n: 0.0 for n in G}
    closeness = nx.closeness_centrality(G)
    df = pd.DataFrame(
        {
            "sensor_name": list(G.nodes()),
            "degree": [degree[n] for n in G.nodes()],
            "betweenness": [betweenness[n] for n in G.nodes()],
            "closeness": [closeness[n] for n in G.nodes()],
        }
    )
    return df


def compute_graph_summary(G: nx.Graph) -> dict[str, float]:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": float(np.mean([deg for _, deg in G.degree()])) if G.number_of_nodes() else 0.0,
    }


def build_temporal_graphs(
    aggregated_df: pd.DataFrame,
    sensor_catalog: pd.DataFrame,
    threshold_km: float = config.ADJACENCY_DISTANCE_THRESHOLD_KM,
    min_coverage: float = 0.0,
) -> list[tuple[pd.Timestamp, nx.Graph]]:
    adjacency, distances = build_adjacency(sensor_catalog, threshold_km)
    graphs: list[tuple[pd.Timestamp, nx.Graph]] = []
    for time_value, df_slice in aggregated_df.groupby("time"):
        coverage = df_slice["sensor_name"].nunique() / sensor_catalog.shape[0]
        if coverage < min_coverage:
            continue
        G = graph_from_catalog(sensor_catalog, adjacency, distances)
        annotate_time_slice(G, df_slice)
        graphs.append((time_value, G))
    return graphs


def compare_graphs(G1: nx.Graph, G2: nx.Graph) -> dict[str, float]:
    e1 = set(G1.edges())
    e2 = set(G2.edges())
    if not e1 and not e2:
        jaccard = 1.0
    else:
        jaccard = len(e1 & e2) / len(e1 | e2)
    return {"edge_jaccard": float(jaccard)}
