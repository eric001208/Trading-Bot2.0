from __future__ import annotations

from collections.abc import Sequence

from crypto_signal_bot.indicators import ma25 as _ma25
from crypto_signal_bot.indicators import ma7 as _ma7
from crypto_signal_bot.indicators import ma99 as _ma99
from crypto_signal_bot.indicators import volume_ma
from crypto_signal_bot.models import Candle, SignalResult

MIN_BARS = 120


def _clamp_score(x: float) -> int:
    return max(0, min(100, int(round(x))))


def evaluate_pullback_long(candles: Sequence[Candle]) -> SignalResult:
    """
    Long-bias pullback: uptrend (above MA99), dip into MA band, reclaim MA25 with volume.

    Scoring is heuristic; intended for alerting thresholds, not execution logic.
    """
    if len(candles) < MIN_BARS:
        return SignalResult("pullback_long", "long", 0, "insufficient_history")

    c = list(candles[-MIN_BARS:])
    last = c[-1]
    prev = c[-2]

    ma7 = _ma7(c)
    ma25v = _ma25(c)
    ma99 = _ma99(c)
    vma = volume_ma(c, 20)

    if ma7 is None or ma25v is None or ma99 is None or vma is None or vma <= 0:
        return SignalResult("pullback_long", "long", 0, "indicators_not_ready")

    score = 0.0
    parts: list[str] = []

    if last.close > ma99:
        dist = (last.close - ma99) / ma99
        t = min(25.0, dist / 0.006 * 25.0)
        score += t
        parts.append(f"trend{t:.0f}")
    else:
        parts.append("no_uptrend")

    lookback = min(8, len(c) - 1)
    recent = c[-lookback - 1 : -1]
    lows = [x.low for x in recent]
    min_low = min(lows) if lows else last.low
    band_lo = min(ma25v, ma7) * 0.997
    band_hi = max(ma25v, ma7) * 1.003
    touched = min_low <= band_hi and min_low >= band_lo * 0.998
    if touched:
        score += 30.0
        parts.append("pullback30")
    else:
        near = min_low <= max(ma25v, ma7) * 1.008
        if near:
            score += 18.0
            parts.append("pullback18")
        else:
            parts.append("no_pullback")

    reclaim = last.close > ma25v and last.close >= last.open
    if reclaim:
        score += 22.0
        parts.append("reclaim22")
    elif last.close > ma25v:
        score += 12.0
        parts.append("reclaim12")

    if last.close >= prev.close:
        score += 8.0
        parts.append("momo8")

    if last.volume >= 0.85 * vma:
        score += 15.0
        parts.append("vol15")
    elif last.volume >= 0.6 * vma:
        score += 8.0
        parts.append("vol8")

    return SignalResult(
        "pullback_long",
        "long",
        _clamp_score(score),
        ";".join(parts),
    )
