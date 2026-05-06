from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable

from crypto_signal_bot.models import Candle, Kline


class CandleStore:
    """
    Thread-safe rolling candles per (symbol, interval).

    - init_from_klines: seed from REST history (closed bars).
    - apply_update: merge live kline websocket events (same open_time updates in place;
      new open_time finalizes the previous bar if still open, then appends).
    """

    def __init__(self, max_len: int) -> None:
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        self._max_len = max_len
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], deque[Candle]] = {}

    @staticmethod
    def _key(symbol: str, interval: str) -> tuple[str, str]:
        return symbol.strip().upper(), interval.strip()

    def init_from_klines(self, symbol: str, interval: str, klines: Iterable[Kline]) -> None:
        key = self._key(symbol, interval)
        dq: deque[Candle] = deque(maxlen=self._max_len)
        for k in klines:
            dq.append(Candle.from_kline(k))
        with self._lock:
            self._data[key] = dq

    def init_from_candles(self, symbol: str, interval: str, candles: Iterable[Candle]) -> None:
        key = self._key(symbol, interval)
        dq: deque[Candle] = deque(maxlen=self._max_len)
        for c in candles:
            dq.append(c.to_closed_copy() if c.is_closed else c)
        with self._lock:
            self._data[key] = dq

    def apply_update(self, symbol: str, interval: str, candle: Candle) -> None:
        key = self._key(symbol, interval)
        with self._lock:
            dq = self._data.get(key)
            if dq is None:
                dq = deque(maxlen=self._max_len)
                self._data[key] = dq
            if not dq:
                dq.append(candle)
                return
            last = dq[-1]
            if candle.open_time > last.open_time:
                if not last.is_closed:
                    dq[-1] = last.to_closed_copy()
                dq.append(candle)
                return
            if candle.open_time == last.open_time:
                dq[-1] = candle
                return

    def snapshot(self, symbol: str, interval: str) -> list[Candle]:
        key = self._key(symbol, interval)
        with self._lock:
            dq = self._data.get(key)
            if not dq:
                return []
            return list(dq)
