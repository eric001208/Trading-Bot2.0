from __future__ import annotations

import asyncio

from crypto_signal_bot.market import binance_futures
from crypto_signal_bot.models import UsdmTicker


def test_fetch_top_usdm_symbols_by_quote_volume_sorts_and_filters(monkeypatch) -> None:
    async def fake_fetch_usdm_24h_tickers(*, base_url: str = "") -> list[UsdmTicker]:
        return [
            UsdmTicker("LOWUSDT", 1.0, 10.0),
            UsdmTicker("BTCUSDT", 1.0, 300.0),
            UsdmTicker("ETHUSDT", 1.0, 200.0),
            UsdmTicker("BTCEUR", 1.0, 999.0),
        ]

    monkeypatch.setattr(binance_futures, "fetch_usdm_24h_tickers", fake_fetch_usdm_24h_tickers)

    symbols = asyncio.run(binance_futures.fetch_top_usdm_symbols_by_quote_volume(limit=2))

    assert symbols == ["BTCUSDT", "ETHUSDT"]
