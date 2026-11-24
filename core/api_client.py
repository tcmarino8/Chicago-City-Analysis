"""Socrata data ingestion with pagination and local caching."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time

import pandas as pd
from sodapy import Socrata

from . import config


def _build_client(app_token: str | None = None) -> Socrata:
    token = app_token or os.getenv(config.APP_TOKEN_ENV_VAR)
    return Socrata(
        domain=config.DATASET_DOMAIN,
        app_token=token,
        timeout=config.SOCRATA_REQUEST_TIMEOUT_SECONDS,
    )


def fetch_rows(
    dataset_id: str = config.DATASET_ID,
    page_size: int = config.PAGE_SIZE,
    limit: int | None = None,
    app_token: str | None = None,
) -> list[dict]:
    """Pull the full dataset via offset-based pagination."""
    client = _build_client(app_token)
    rows: list[dict] = []
    offset = 0
    remaining = limit
    while True:
        batch_size = page_size if remaining is None else min(page_size, remaining)
        data = None
        for attempt in range(config.SOCRATA_MAX_RETRIES + 1):
            try:
                data = client.get(dataset_id, limit=batch_size, offset=offset)
                break
            except Exception:
                if attempt >= config.SOCRATA_MAX_RETRIES:
                    raise
                delay = config.SOCRATA_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(delay)

        if data is None:
            break
        if not data:
            break
        rows.extend(data)
        offset += len(data)
        if remaining is not None:
            remaining -= len(data)
            if remaining <= 0:
                break
        if len(data) < batch_size:
            break
    return rows


def fetch_dataframe(**kwargs) -> pd.DataFrame:
    """Fetch Socrata rows and convert to a DataFrame."""
    rows = fetch_rows(**kwargs)
    return pd.DataFrame.from_records(rows)


def load_data(
    use_cache: bool = True,
    refresh: bool = False,
    cache_path: str | Path = config.CACHE_CSV,
    prefer_live: bool | None = None,
    **fetch_kwargs,
) -> pd.DataFrame:
    """Read live Socrata data first in production, with a resilient cache fallback."""
    cache_file = Path(cache_path)
    if prefer_live is None:
        live_setting = os.getenv(config.PREFER_LIVE_DATA_ENV_VAR, "").strip().lower()
        prefer_live = live_setting in {"1", "true", "yes", "on"} or bool(os.getenv("VERCEL"))

    if prefer_live or refresh:
        fetch_kwargs.setdefault("order", f"{config.DEFAULT_TIME_COLUMN} DESC")
        fetch_kwargs.setdefault("limit", config.SOCRATA_LIVE_ROW_LIMIT)
        try:
            df = fetch_dataframe(**fetch_kwargs)
            if df.empty:
                raise RuntimeError("Socrata returned no rows.")
            df.attrs["source"] = "live"
            df.attrs["refreshed_at"] = datetime.now(timezone.utc).isoformat()
            if use_cache and not bool(os.getenv("VERCEL")):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file, index=False)
            return df
        except Exception as exc:
            if not (use_cache and cache_file.exists()):
                raise RuntimeError(f"Live Socrata request failed and no local cache is available: {exc}") from exc
            df = pd.read_csv(cache_file)
            df.attrs["source"] = "cache-fallback"
            df.attrs["refreshed_at"] = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc).isoformat()
            df.attrs["source_warning"] = str(exc)
            return df

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        df.attrs["source"] = "cache"
        df.attrs["refreshed_at"] = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc).isoformat()
        df.attrs["source_warning"] = None
        return df

    df = fetch_dataframe(**fetch_kwargs)
    df.attrs["source"] = "live"
    df.attrs["refreshed_at"] = datetime.now(timezone.utc).isoformat()
    df.attrs["source_warning"] = None
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
    return df
