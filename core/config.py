"""Central configuration for the Chicago AQI Sensor Network project."""
from __future__ import annotations

from pathlib import Path

# --- Data & API settings ---------------------------------------------------
DATASET_DOMAIN: str = "data.cityofchicago.org"
DATASET_ID: str = "xfya-dxtq"
DEFAULT_AQI_COLUMN: str = "pm2_5concmassindividual_value"
DEFAULT_TIME_COLUMN: str = "time"
DEFAULT_LAT_COLUMN: str = "latitude"
DEFAULT_LON_COLUMN: str = "longitude"
DEFAULT_SENSOR_CANDIDATES: tuple[str, ...] = (
    "sensor_name",
    "sensor",
    "sensor_id",
    "name",
    "site_id",
)

# API pagination (Socrata allows up to 50k rows per request for anonymous apps)
PAGE_SIZE: int = 50000
APP_TOKEN_ENV_VAR: str = "SOCRATA_APP_TOKEN"

# --- Local storage ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_CSV = DATA_DIR / "chicago_aqi_cache.csv"

# --- Visualization defaults ------------------------------------------------
BIN_OPTIONS: tuple[str, ...] = ("15min", "1h", "3h", "6h", "12h", "1d", "3d", "1w")
COVERAGE_THRESHOLDS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.6, 0.75)
DEFAULT_BIN: str = "12h"
DEFAULT_COVERAGE_THRESHOLD: float = 0.5
DEFAULT_GRID_RESOLUTION: int = 80
DEFAULT_MAX_DISTANCE_KM: float = 5.0
ADJACENCY_DISTANCE_THRESHOLD_KM: float = 10.0

# --- NOAA weather overlay --------------------------------------------------
WEATHER_BASE_URL: str = "https://api.weather.gov"
WEATHER_USER_AGENT: str = "EODataChicagoAQI/1.0 (contact: local-workspace-user)"
WEATHER_ACCEPT: str = "application/geo+json"
WEATHER_REQUEST_TIMEOUT_SECONDS: float = 15.0
WEATHER_STATION_LIMIT: int = 10
WEATHER_POINT_SAMPLE_LIMIT: int = 5

# --- Misc ------------------------------------------------------------------
EXPORT_HTML_DIR = PROJECT_ROOT
