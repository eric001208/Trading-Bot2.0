from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from crypto_signal_bot.market import fetch_usdm_klines


def test_fetch_btcusdt_15m_live() -> None:
    """Integration test against Binance public FAPI (requires network)."""
    base = os.getenv("BINANCE_FUTURES_REST_BASE", "https://fapi.binance.com").rstrip("/")
    try:
        klines = asyncio.run(
            fetch_usdm_klines(
                symbol="BTCUSDT",
                interval="15m",
                limit=10,
                base_url=base,
            )
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 451:
            pytest.skip("Binance FAPI is unavailable from this network location.")
        raise
    assert len(klines) == 10
    last = klines[-1]
    assert last.open > 0 and last.high >= last.low
    assert last.close > 0 and last.volume >= 0
    assert last.close_time >= last.open_time
