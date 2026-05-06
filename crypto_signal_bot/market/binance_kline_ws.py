from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence

import websockets
from websockets.exceptions import ConnectionClosed

from crypto_signal_bot.candle_store import CandleStore
from crypto_signal_bot.market.binance_kline_stream import (
    DEFAULT_FSTREAM_WS_BASE,
    build_combined_kline_url,
    loads_ws_json,
    parse_kline_ws_payload,
)
from crypto_signal_bot.models import Candle

logger = logging.getLogger(__name__)

OnCandle = Callable[[str, str, Candle], Awaitable[None] | None]


async def consume_usdm_klines(
    *,
    store: CandleStore,
    symbols: Sequence[str],
    intervals: Sequence[str],
    ws_base: str = DEFAULT_FSTREAM_WS_BASE,
    on_candle: OnCandle | None = None,
    reconnect_delay_s: float = 1.0,
) -> None:
    """
    Subscribe to USD-M futures kline streams and merge updates into ``store``.

    Runs until cancelled. Reconnects on disconnect. Calls ``on_candle`` after each
    ``store.apply_update`` (sync or async).
    """
    url = build_combined_kline_url(ws_base, symbols, intervals)
    logger.info("Binance kline WS connecting: %s", url[:96] + ("..." if len(url) > 96 else ""))

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                async for raw in ws:
                    payload = loads_ws_json(raw)
                    if payload is None:
                        continue
                    parsed = parse_kline_ws_payload(payload)
                    if parsed is None:
                        continue
                    sym, interval, candle = parsed
                    store.apply_update(sym, interval, candle)
                    if on_candle is not None:
                        out = on_candle(sym, interval, candle)
                        if inspect.isawaitable(out):
                            await out
        except ConnectionClosed:
            logger.warning("websocket closed; reconnecting")
        except OSError as e:
            logger.warning("websocket error %s; reconnecting", e)
        await asyncio.sleep(reconnect_delay_s)
