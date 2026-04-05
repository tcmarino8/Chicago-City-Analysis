"""Socrata data ingestion with pagination and local caching."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from sodapy import Socrata

from . import config


def _build_client(app_token: str | None = None) -> Socrata:
    token = app_token or os.getenv(config.APP_TOKEN_ENV_VAR)
    return Socrata(domain=config.DATASET_DOMAIN, app_token=token)


def fetch_rows(
    dataset_id: str = config.DATASET_ID,
    page_size: int = config.PAGE_SIZE,
    limit: int | None = None,
    app_token: str | None = None,
    **query_kwargs: Any,
) -> list[dict[str, Any]]:
    """Pull the full dataset via offset-based pagination."""
    client = _build_client(app_token)
    rows: list[dict[str, Any]] = []
    offset = 0
    remaining = limit
    request_kwargs = {key: value for key, value in query_kwargs.items() if value is not None}
    while True:
        batch_size = page_size if remaining is None else min(page_size, remaining)
        data = client.get(dataset_id, limit=batch_size, offset=offset, **request_kwargs)
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
    **fetch_kwargs,
) -> pd.DataFrame:
    """Read from cache or fetch remote data as needed."""
    cache_file = Path(cache_path)

    if use_cache and not refresh and cache_file.exists():
        return pd.read_csv(cache_file)

    df = fetch_dataframe(**fetch_kwargs)
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
    return df
