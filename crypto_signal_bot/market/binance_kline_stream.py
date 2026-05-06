from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from crypto_signal_bot.models import Candle

logger = logging.getLogger(__name__)

DEFAULT_FSTREAM_WS_BASE = "wss://fstream.binance.com"


def build_combined_kline_url(
    ws_base: str,
    symbols: Sequence[str],
    intervals: Sequence[str],
) -> str:
    """
    USD-M futures combined stream: /stream?streams=btcusdt@kline_15m/ethusdt@kline_1h
    """
    streams: list[str] = []
    for s in symbols:
        sym = s.strip().lower()
        for iv in intervals:
            streams.append(f"{sym}@kline_{iv.strip()}")
    path = "/stream?streams=" + "/".join(streams)
    return f"{ws_base.rstrip('/')}{path}"


def parse_kline_ws_payload(payload: dict[str, Any]) -> tuple[str, str, Candle] | None:
    """
    Parse Binance futures kline event (combined or single wrapper).

    Returns (symbol_upper, interval, candle) or None if not a kline event.
    """
    data = payload.get("data")
    if data is None and payload.get("e") == "kline":
        data = payload
    if not isinstance(data, dict) or data.get("e") != "kline":
        return None
    sym = str(data.get("s", "")).strip().upper()
    k = data.get("k")
    if not sym or not isinstance(k, dict):
        return None
    interval = str(k.get("i", "")).strip()
    if not interval:
        return None
    try:
        candle = Candle(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=int(k["T"]),
            is_closed=bool(k.get("x", False)),
        )
    except (KeyError, TypeError, ValueError):
        logger.debug("malformed kline payload: %s", payload, exc_info=True)
        return None
    return sym, interval, candle


def loads_ws_json(raw: str | bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None
