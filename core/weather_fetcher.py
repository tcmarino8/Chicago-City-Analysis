"""Live NOAA weather ingestion for workspace overlays."""
from __future__ import annotations

from functools import lru_cache
import json
from math import atan2, cos, radians, sin, sqrt
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config, geo_utils


def _request_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": config.WEATHER_USER_AGENT,
            "Accept": config.WEATHER_ACCEPT,
        },
    )
    with urlopen(request, timeout=config.WEATHER_REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


@lru_cache(maxsize=64)
def _get_points_metadata(lat: float, lon: float) -> dict[str, Any]:
    return _request_json(f"{config.WEATHER_BASE_URL}/points/{lat:.4f},{lon:.4f}")


@lru_cache(maxsize=64)
def _get_station_listing(stations_url: str) -> dict[str, Any]:
    return _request_json(stations_url)


@lru_cache(maxsize=256)
def _get_latest_observation(station_id: str) -> dict[str, Any]:
    return _request_json(f"{config.WEATHER_BASE_URL}/stations/{station_id}/observations/latest")


def _value(properties: dict[str, Any], key: str) -> float | None:
    item = properties.get(key) or {}
    value = item.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2.0) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * atan2(sqrt(a), sqrt(max(1.0 - a, 0.0)))


def _wind_cardinal(direction_deg: float | None) -> str | None:
    if direction_deg is None:
        return None
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[round(direction_deg / 45.0) % len(labels)]


def _cloud_info(properties: dict[str, Any]) -> tuple[list[str], bool]:
    layers = properties.get("cloudLayers") or []
    amounts = [layer.get("amount") for layer in layers if layer.get("amount")]
    description = (properties.get("textDescription") or "").lower()
    cloudy_terms = ("cloud", "overcast", "broken", "scattered", "few", "fog", "haze", "mist")
    has_clouds = any(amount not in {"CLR", "SKC"} for amount in amounts) or any(term in description for term in cloudy_terms)
    return amounts, has_clouds


def _sample_points(bbox: geo_utils.BoundingBox, center: dict[str, float]) -> list[tuple[float, float]]:
    points = [
        (center["lat"], center["lon"]),
        (bbox.lat_min, bbox.lon_min),
        (bbox.lat_min, bbox.lon_max),
        (bbox.lat_max, bbox.lon_min),
        (bbox.lat_max, bbox.lon_max),
    ]
    unique_points: list[tuple[float, float]] = []
    for lat, lon in points:
        rounded = (round(float(lat), 4), round(float(lon), 4))
        if rounded not in unique_points:
            unique_points.append(rounded)
    return unique_points[: config.WEATHER_POINT_SAMPLE_LIMIT]


def _station_candidates(bbox: geo_utils.BoundingBox, center: dict[str, float]) -> list[dict[str, Any]]:
    stations_by_id: dict[str, dict[str, Any]] = {}
    for lat, lon in _sample_points(bbox, center):
        try:
            points_data = _get_points_metadata(lat, lon)
            stations_url = points_data.get("properties", {}).get("observationStations")
            if not stations_url:
                continue
            listing = _get_station_listing(stations_url)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        for feature in listing.get("features", []):
            props = feature.get("properties", {})
            station_id = str(props.get("stationIdentifier") or "").strip()
            geometry = feature.get("geometry", {}) or {}
            coords = geometry.get("coordinates") or []
            if not station_id or len(coords) < 2:
                continue
            station_lon = float(coords[0])
            station_lat = float(coords[1])
            distance_km = _haversine_km(center["lat"], center["lon"], station_lat, station_lon)
            existing = stations_by_id.get(station_id)
            if existing is None or distance_km < existing["distance_to_center_km"]:
                stations_by_id[station_id] = {
                    "station_id": station_id,
                    "name": props.get("name") or station_id,
                    "latitude": station_lat,
                    "longitude": station_lon,
                    "distance_to_center_km": distance_km,
                }
    return sorted(stations_by_id.values(), key=lambda item: item["distance_to_center_km"])[: config.WEATHER_STATION_LIMIT]


def _normalize_observation(station_meta: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    props = observation.get("properties", {}) or {}
    cloud_amounts, has_clouds = _cloud_info(props)
    temperature_c = _value(props, "temperature")
    wind_speed_ms = _value(props, "windSpeed")
    wind_direction_deg = _value(props, "windDirection")
    pressure_pa = _value(props, "barometricPressure")
    return {
        "station_id": station_meta["station_id"],
        "name": station_meta["name"],
        "latitude": station_meta["latitude"],
        "longitude": station_meta["longitude"],
        "distance_to_center_km": station_meta["distance_to_center_km"],
        "timestamp": props.get("timestamp"),
        "temperature_c": temperature_c,
        "temperature_f": None if temperature_c is None else (temperature_c * 9.0 / 5.0) + 32.0,
        "wind_speed_m_s": wind_speed_ms,
        "wind_speed_mph": None if wind_speed_ms is None else wind_speed_ms * 2.23694,
        "wind_direction_deg": wind_direction_deg,
        "wind_direction_cardinal": _wind_cardinal(wind_direction_deg),
        "pressure_pa": pressure_pa,
        "pressure_hpa": None if pressure_pa is None else pressure_pa / 100.0,
        "cloud_amounts": cloud_amounts,
        "has_clouds": has_clouds,
        "text_description": props.get("textDescription"),
    }


def fetch_workspace_weather(bbox: geo_utils.BoundingBox, center: dict[str, float]) -> dict[str, Any]:
    try:
        stations = _station_candidates(bbox, center)
        points: list[dict[str, Any]] = []
        failures = 0
        for station in stations:
            try:
                obs = _get_latest_observation(station["station_id"])
                points.append(_normalize_observation(station, obs))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
                failures += 1
        latest_timestamps = [point["timestamp"] for point in points if point.get("timestamp")]
        summary = [
            f"Weather stations loaded: {len(points)}",
            f"Request failures: {failures}",
        ]
        if latest_timestamps:
            summary.append(f"Latest observation: {max(latest_timestamps)}")
        cloudy_count = sum(1 for point in points if point.get("has_clouds"))
        summary.append(f"Cloudy / obstructed stations: {cloudy_count}")
        wind_values = [point["wind_speed_mph"] for point in points if point.get("wind_speed_mph") is not None]
        if wind_values:
            summary.append(f"Mean wind speed: {sum(wind_values) / len(wind_values):.1f} mph")
        temp_values = [point["temperature_f"] for point in points if point.get("temperature_f") is not None]
        if temp_values:
            summary.append(f"Mean temperature: {sum(temp_values) / len(temp_values):.1f} °F")
        return {
            "available": bool(points),
            "points": points,
            "summary": summary,
            "loaded": True,
        }
    except Exception as exc:  # pragma: no cover - network safety
        return {
            "available": False,
            "points": [],
            "summary": [f"Weather overlay unavailable: {exc}"],
            "loaded": True,
        }
