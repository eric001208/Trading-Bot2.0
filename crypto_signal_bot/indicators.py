from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from crypto_signal_bot.models import Candle

DEFAULT_VOLUME_MA_PERIOD = 20


def sma(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    seed = sma(values[:period], period)
    if seed is None:
        return None
    e = seed
    for v in values[period:]:
        e = (v - e) * k + e
    return e


def ema_slope(values: Sequence[float], period: int, lookback: int = 3) -> float | None:
    if lookback <= 0 or period <= 0 or len(values) < period + lookback:
        return None
    now = ema(values, period)
    prev = ema(values[: -lookback], period)
    if now is None or prev is None:
        return None
    return now - prev


def closes(candles: Sequence[Candle]) -> list[float]:
    return [c.close for c in candles]


def volumes(candles: Sequence[Candle]) -> list[float]:
    return [c.volume for c in candles]


def ma7(candles: Sequence[Candle]) -> float | None:
    return sma(closes(candles), 7)


def ma25(candles: Sequence[Candle]) -> float | None:
    return sma(closes(candles), 25)


def ma99(candles: Sequence[Candle]) -> float | None:
    return sma(closes(candles), 99)


def volume_ma(candles: Sequence[Candle], period: int = DEFAULT_VOLUME_MA_PERIOD) -> float | None:
    return sma(volumes(candles), period)


@dataclass(frozen=True)
class MovingAverages:
    ma7: float | None
    ma25: float | None
    ma99: float | None
    volume_ma: float | None


def compute_moving_averages(
    candles: Sequence[Candle],
    *,
    volume_ma_period: int = DEFAULT_VOLUME_MA_PERIOD,
) -> MovingAverages:
    """MA7 / MA25 / MA99 on close; volume MA uses `volume_ma_period` (default 20)."""
    return MovingAverages(
        ma7=ma7(candles),
        ma25=ma25(candles),
        ma99=ma99(candles),
        volume_ma=volume_ma(candles, volume_ma_period),
    )
