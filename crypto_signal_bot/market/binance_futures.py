from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from crypto_signal_bot.models import Kline, UsdmTicker

DEFAULT_FAPI_BASE_URL = "https://fapi.binance.com"
DEFAULT_PUBLIC_DATA_BASE_URL = "https://data.binance.vision"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _parse_kline_row(row: list[Any]) -> Kline:
    return Kline(
        open_time=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=int(row[6]),
    )


def _parse_ticker_row(row: dict[str, Any]) -> UsdmTicker:
    return UsdmTicker(
        symbol=str(row["symbol"]).strip().upper(),
        last_price=float(row["lastPrice"]),
        quote_volume=float(row["quoteVolume"]),
    )


async def fetch_usdm_klines(
    *,
    symbol: str,
    interval: str,
    limit: int = 500,
    base_url: str = DEFAULT_FAPI_BASE_URL,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> list[Kline]:
    """
    Binance USD-M Futures public klines (GET /fapi/v1/klines).

    Docs: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
    """
    url = f"{base_url.rstrip('/')}/fapi/v1/klines"
    sym = symbol.strip().upper()
    params: dict[str, str | int] = {"symbol": sym, "interval": interval.strip(), "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        raw = r.json()

    if not isinstance(raw, list):
        raise TypeError(f"unexpected klines payload: {type(raw)}")

    return [_parse_kline_row(row) for row in raw]


async def fetch_usdm_klines_range(
    *,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    base_url: str = DEFAULT_FAPI_BASE_URL,
    limit: int = 1000,
) -> list[Kline]:
    """Fetch historical klines over a time range by paging Binance's kline endpoint."""
    step_ms = INTERVAL_MS.get(interval.strip())
    if step_ms is None:
        raise ValueError(f"unsupported interval for range fetch: {interval}")
    if start_time_ms >= end_time_ms:
        return []

    out: list[Kline] = []
    cursor = start_time_ms
    seen: set[int] = set()
    while cursor < end_time_ms:
        try:
            batch = await fetch_usdm_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                base_url=base_url,
                start_time_ms=cursor,
                end_time_ms=end_time_ms,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 451 and base_url.rstrip("/") == DEFAULT_FAPI_BASE_URL:
                return await fetch_public_usdm_klines_range(
                    symbol=symbol,
                    interval=interval,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )
            raise
        if not batch:
            break
        for kline in batch:
            if kline.open_time not in seen and kline.open_time < end_time_ms:
                out.append(kline)
                seen.add(kline.open_time)
        last_open = batch[-1].open_time
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < limit:
            break

    return sorted(out, key=lambda k: k.open_time)


def _utc_dates_between(start_time_ms: int, end_time_ms: int) -> list[date]:
    start = datetime.fromtimestamp(start_time_ms / 1000, UTC).date()
    end = datetime.fromtimestamp((end_time_ms - 1) / 1000, UTC).date()
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


async def _fetch_public_usdm_klines_day(
    *,
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    day: date,
    base_url: str,
) -> list[Kline]:
    sym = symbol.strip().upper()
    interval = interval.strip()
    day_text = day.isoformat()
    url = (
        f"{base_url.rstrip('/')}/data/futures/um/daily/klines/"
        f"{sym}/{interval}/{sym}-{interval}-{day_text}.zip"
    )
    response = await client.get(url)
    if response.status_code == 404:
        return []
    response.raise_for_status()

    out: list[Kline] = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
        if not csv_names:
            return []
        with zf.open(csv_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for row in csv.reader(text):
                if not row:
                    continue
                try:
                    out.append(_parse_kline_row(row))
                except (ValueError, TypeError):
                    continue
    return out


async def fetch_public_usdm_klines_range(
    *,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    base_url: str = DEFAULT_PUBLIC_DATA_BASE_URL,
) -> list[Kline]:
    """Fetch USD-M futures klines from Binance's public zip archive."""
    if INTERVAL_MS.get(interval.strip()) is None:
        raise ValueError(f"unsupported interval for public data fetch: {interval}")
    if start_time_ms >= end_time_ms:
        return []

    out: list[Kline] = []
    seen: set[int] = set()
    async with httpx.AsyncClient(timeout=60.0) as client:
        for day in _utc_dates_between(start_time_ms, end_time_ms):
            daily = await _fetch_public_usdm_klines_day(
                client=client,
                symbol=symbol,
                interval=interval,
                day=day,
                base_url=base_url,
            )
            for kline in daily:
                if start_time_ms <= kline.open_time < end_time_ms and kline.open_time not in seen:
                    out.append(kline)
                    seen.add(kline.open_time)
    return sorted(out, key=lambda k: k.open_time)


async def fetch_usdm_24h_tickers(
    *,
    base_url: str = DEFAULT_FAPI_BASE_URL,
) -> list[UsdmTicker]:
    """Binance USD-M Futures 24hr ticker statistics."""
    url = f"{base_url.rstrip('/')}/fapi/v1/ticker/24hr"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        # Some regions return HTTP 451 for this endpoint. For backtests and scanners
        # we can safely degrade to an empty list (dynamic pools will be skipped).
        if r.status_code == 451 and base_url.rstrip("/") == DEFAULT_FAPI_BASE_URL:
            return []
        r.raise_for_status()
        raw = r.json()

    if not isinstance(raw, list):
        raise TypeError(f"unexpected 24hr ticker payload: {type(raw)}")

    tickers: list[UsdmTicker] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            ticker = _parse_ticker_row(row)
        except (KeyError, TypeError, ValueError):
            continue
        tickers.append(ticker)
    return tickers


async def fetch_top_usdm_symbols_by_quote_volume(
    *,
    limit: int = 10,
    base_url: str = DEFAULT_FAPI_BASE_URL,
) -> list[str]:
    if limit <= 0:
        return []
    tickers = await fetch_usdm_24h_tickers(base_url=base_url)
    ranked = sorted(
        (t for t in tickers if t.symbol.endswith("USDT") and t.quote_volume > 0),
        key=lambda t: t.quote_volume,
        reverse=True,
    )
    return [t.symbol for t in ranked[:limit]]
