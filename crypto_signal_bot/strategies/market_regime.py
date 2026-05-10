from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from crypto_signal_bot.indicators import closes, ma25, ma99
from crypto_signal_bot.models import Candle

DEFAULT_ENTRY_RANGE_LOOKBACK_BARS = 32  # ~8 hours on 15m


@dataclass(frozen=True)
class MarketRegimeResult:
    regime: str  # "trend" / "range" / "compression" / "unknown"
    confidence: int  # 0-100
    reasons: tuple[str, ...]
    no_trade_zone: bool


def _avg_true_range(candles: Sequence[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    ranges: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        ranges.append(max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close)))
    return sum(ranges) / period


def _adx(candles: Sequence[Candle], period: int = 14) -> float | None:
    """
    Simple DX proxy in [0, 100] based on the last `period` bars.
    We keep it intentionally lightweight and self-contained (no external libs).
    """
    if len(candles) < period + 1:
        return None

    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        current = candles[i]
        prev = candles[i - 1]
        up_move = current.high - prev.high
        down_move = prev.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(current.high - current.low, abs(current.high - prev.close), abs(current.low - prev.close)))

    tr_sum = sum(trs)
    if tr_sum <= 0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm) / tr_sum
    minus_di = 100.0 * sum(minus_dm) / tr_sum
    denom = plus_di + minus_di
    if denom <= 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / denom


def _ma25_slope(candles: Sequence[Candle]) -> float | None:
    # Same heuristic as observation strategy: compare two 25-bar windows.
    if len(candles) < 50:
        return None
    prev = sum(c.close for c in candles[-50:-25]) / 25
    now = sum(c.close for c in candles[-25:]) / 25
    return now - prev


def _atr_series(candles: Sequence[Candle], period: int = 14, max_points: int = 120) -> list[float]:
    if period <= 0 or len(candles) < period + 1:
        return []
    start = max(period + 1, len(candles) - max_points)
    out: list[float] = []
    for end in range(start, len(candles) + 1):
        atr = _avg_true_range(candles[:end], period)
        if atr is not None:
            out.append(atr)
    return out


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    mid = len(arr) // 2
    if len(arr) % 2 == 1:
        return float(arr[mid])
    return 0.5 * (arr[mid - 1] + arr[mid])


def detect_market_regime(
    *,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    entry_range_lookback_bars: int = DEFAULT_ENTRY_RANGE_LOOKBACK_BARS,
) -> MarketRegimeResult:
    """
    Market regime classification for breakout-style signals.

    Outputs:
      - regime: "trend" / "range" / "compression" / "unknown"
      - confidence: 0-100
      - reasons: human-readable details
      - no_trade_zone: True when price is in the middle 40%-60% of recent range.
    """
    reasons: list[str] = []
    missing: list[str] = []

    if not trend_candles:
        missing.append("trend_candles")
    if not entry_candles:
        missing.append("entry_candles")
    if missing:
        return MarketRegimeResult(
            regime="unknown",
            confidence=0,
            reasons=tuple([f"数据缺失：{', '.join(missing)}"]),
            no_trade_zone=False,
        )

    price = entry_candles[-1].close
    if price <= 0:
        return MarketRegimeResult("unknown", 0, ("价格非法，无法判断市场状态。",), False)

    # --- no-trade zone (range mid 40%-60%) ---
    no_trade_zone = False
    if len(entry_candles) >= max(5, int(entry_range_lookback_bars)):
        recent = entry_candles[-int(entry_range_lookback_bars) :]
        hi = max(c.high for c in recent)
        lo = min(c.low for c in recent)
        if hi > lo:
            pos = (price - lo) / (hi - lo)
            reasons.append(f"近区间位置：pos={(pos*100):.1f}%（区间 {lo:g}-{hi:g}）")
            if 0.40 <= pos <= 0.60:
                no_trade_zone = True
                reasons.append("no_trade_zone：价格处于区间中部 40%-60%，避免在震荡中间追突破。")
        else:
            reasons.append("近区间高低点重合，无法计算区间位置。")
    else:
        reasons.append("K线数量不足，无法计算 no_trade_zone 区间位置。")

    # Indicators (requirement: do not fabricate when unavailable).
    adx_15m = _adx(entry_candles, 14)
    atr_15m = _avg_true_range(entry_candles, 14)
    atr_1h = _avg_true_range(trend_candles, 14)
    m25 = ma25(trend_candles)
    m99 = ma99(trend_candles)
    slope = _ma25_slope(trend_candles)

    if adx_15m is None:
        missing.append("adx_15m")
    if atr_15m is None:
        missing.append("atr_15m")
    if atr_1h is None:
        missing.append("atr_1h")
    if m25 is None:
        missing.append("ma25_1h")
    if m99 is None:
        missing.append("ma99_1h")
    if slope is None:
        missing.append("ma25_slope_1h")

    if missing:
        return MarketRegimeResult(
            regime="unknown",
            confidence=0,
            reasons=tuple(reasons + [f"数据不足：{', '.join(missing)}"]),
            no_trade_zone=no_trade_zone,
        )

    assert adx_15m is not None
    assert atr_15m is not None
    assert atr_1h is not None
    assert m25 is not None
    assert m99 is not None
    assert slope is not None

    # Range pct over recent entry window (used in compression scoring).
    range_pct = 0.0
    if len(entry_candles) >= max(5, int(entry_range_lookback_bars)):
        recent = entry_candles[-int(entry_range_lookback_bars) :]
        hi = max(c.high for c in recent)
        lo = min(c.low for c in recent)
        if price > 0 and hi > lo:
            range_pct = (hi - lo) / price
    reasons.append(f"15m ADX={adx_15m:.1f} ATR%={(atr_15m/price*100):.3f}% range%={(range_pct*100):.3f}%")

    # Compression score: ATR is low relative to its own recent baseline + range is narrow.
    atr_values = _atr_series(entry_candles, 14, max_points=120)
    atr_med = _median(atr_values)
    atr_ratio = (atr_15m / atr_med) if (atr_med is not None and atr_med > 0) else None
    if atr_ratio is None:
        reasons.append("ATR baseline 不足，无法计算压缩强度。")
    else:
        reasons.append(f"ATR压缩比：{atr_ratio:.2f}（当前ATR / 近期ATR中位数）")

    compression_score = 0.0
    if atr_ratio is not None:
        if atr_ratio <= 0.70:
            compression_score += 60.0
        elif atr_ratio <= 0.80:
            compression_score += 45.0
        elif atr_ratio <= 0.90:
            compression_score += 25.0
    if range_pct > 0:
        if range_pct <= 0.012:
            compression_score += 20.0
        elif range_pct <= 0.018:
            compression_score += 10.0
    if adx_15m <= 14:
        compression_score += 10.0

    # Trend score: 1h MA alignment + slope + price location.
    close_1h = trend_candles[-1].close
    up = close_1h >= m25 >= m99 and slope > 0
    down = close_1h <= m25 <= m99 and slope < 0
    trend_score = 0.0
    if up or down:
        trend_score += 60.0
        reasons.append("1h均线结构明确：价格沿MA25/MA99同向运行。")
    else:
        reasons.append("1h均线结构不够明确。")
    # Penalize overly extended conditions (breakouts in the middle of a stretch are noisier).
    if atr_1h > 0:
        dist_atr = abs(close_1h - m25) / atr_1h
        reasons.append(f"1h距离MA25：{dist_atr:.2f} ATR")
        if dist_atr <= 2.0:
            trend_score += 10.0
        elif dist_atr <= 3.0:
            trend_score += 5.0
        else:
            trend_score -= 5.0
    # Use slope magnitude as strength proxy.
    if atr_1h > 0:
        slope_ratio = abs(slope) / atr_1h
        reasons.append(f"1h MA25斜率强度：{slope_ratio:.2f} ATR/25bars")
        if slope_ratio >= 0.45:
            trend_score += 20.0
        elif slope_ratio >= 0.25:
            trend_score += 12.0
        elif slope_ratio >= 0.15:
            trend_score += 6.0
    # ADX supports trend confidence.
    if adx_15m >= 25:
        trend_score += 10.0
    elif adx_15m >= 18:
        trend_score += 6.0

    # Range score: low ADX + weak slope + mean-reverting position.
    range_score = 0.0
    if adx_15m <= 12:
        range_score += 55.0
    elif adx_15m <= 14:
        range_score += 40.0
    elif adx_15m <= 16:
        range_score += 25.0
    if atr_1h > 0:
        slope_ratio = abs(slope) / atr_1h
        if slope_ratio <= 0.15:
            range_score += 25.0
        elif slope_ratio <= 0.22:
            range_score += 15.0
    if no_trade_zone:
        range_score += 10.0

    # Decide regime (prioritize compression > trend > range).
    if compression_score >= 70.0:
        conf = int(min(100.0, compression_score))
        return MarketRegimeResult(
            regime="compression",
            confidence=conf,
            reasons=tuple(reasons + ["判定：波动压缩，等待有效突破确认。"]),
            no_trade_zone=no_trade_zone,
        )
    if trend_score >= 70.0:
        conf = int(min(100.0, trend_score))
        return MarketRegimeResult(
            regime="trend",
            confidence=conf,
            reasons=tuple(reasons + ["判定：趋势市，允许顺势突破/回踩。"]),
            no_trade_zone=no_trade_zone,
        )
    if range_score >= 60.0:
        conf = int(min(100.0, range_score))
        return MarketRegimeResult(
            regime="range",
            confidence=conf,
            reasons=tuple(reasons + ["判定：震荡市，避免在区间中部频繁做突破交易。"]),
            no_trade_zone=no_trade_zone,
        )

    # Fallback: ambiguous state.
    return MarketRegimeResult(
        regime="unknown",
        confidence=20,
        reasons=tuple(reasons + ["判定：状态不明确（unknown），保守起见不建议交易。"]),
        no_trade_zone=no_trade_zone,
    )

