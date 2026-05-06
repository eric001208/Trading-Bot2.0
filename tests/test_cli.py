from __future__ import annotations

from crypto_signal_bot.cli import (
    _dynamic_alt_candidates_for_days,
    _latest_dynamic_alt_candidate,
    _parse_direction_thresholds,
    _threshold_for_candidate,
)
from crypto_signal_bot.models import Candle


def _daily_bar(day: int, open_price: float, close: float, volume: float) -> Candle:
    open_time = day * 86_400_000
    high = max(open_price, close) * 1.04
    low = min(open_price, close) * 0.97
    return Candle(open_time, open_price, high, low, close, volume, open_time + 86_400_000 - 1, True)


def test_parse_direction_thresholds_accepts_direction_aliases() -> None:
    thresholds = _parse_direction_thresholds("BTCUSDT:short:98,ETHUSDT:long:96,SOLUSDT:做空:100")

    assert thresholds[("BTCUSDT", "做空观察")] == 98
    assert thresholds[("ETHUSDT", "做多观察")] == 96
    assert thresholds[("SOLUSDT", "做空观察")] == 100


def test_direction_threshold_overrides_symbol_threshold() -> None:
    symbol_thresholds = {"BTCUSDT": 92}
    direction_thresholds = {("BTCUSDT", "做空观察"): 98}

    assert _threshold_for_candidate("BTCUSDT", "做空观察", 90, symbol_thresholds, direction_thresholds) == 98
    assert _threshold_for_candidate("BTCUSDT", "做多观察", 90, symbol_thresholds, direction_thresholds) == 92


def test_latest_dynamic_alt_candidate_requires_volume_surge() -> None:
    daily = [_daily_bar(day, 10.0, 10.2, 100.0) for day in range(7)]
    daily.append(_daily_bar(7, 10.0, 10.6, 230.0))

    candidate = _latest_dynamic_alt_candidate(
        symbol="DOGEUSDT",
        daily_candles=daily,
        as_of_ms=daily[-1].close_time + 1,
        lookback_days=7,
        min_volume_ratio=1.5,
        min_quote_volume=0.0,
        min_daily_move_pct=0.0,
        max_daily_move_pct=0.25,
        min_daily_range_pct=0.03,
        max_daily_range_pct=0.18,
    )

    assert candidate is not None
    assert candidate.symbol == "DOGEUSDT"
    assert candidate.volume_ratio > 1.5


def test_dynamic_alt_candidates_for_days_limits_entry_day() -> None:
    daily = [_daily_bar(day, 10.0, 10.2, 100.0) for day in range(7)]
    daily.append(_daily_bar(7, 10.0, 10.6, 230.0))

    candidates = _dynamic_alt_candidates_for_days(
        symbol="DOGEUSDT",
        daily_candles=daily,
        start_day="1970-01-09",
        end_day="1970-01-09",
        lookback_days=7,
        min_volume_ratio=1.5,
        min_quote_volume=0.0,
        min_daily_move_pct=0.0,
        max_daily_move_pct=0.25,
        min_daily_range_pct=0.03,
        max_daily_range_pct=0.18,
    )

    assert len(candidates) == 1
    assert candidates[0].entry_day == "1970-01-09"


def test_dynamic_alt_candidate_filters_down_volume_spike_by_default() -> None:
    daily = [_daily_bar(day, 10.0, 10.2, 100.0) for day in range(7)]
    daily.append(_daily_bar(7, 10.0, 9.5, 260.0))

    candidate = _latest_dynamic_alt_candidate(
        symbol="DOGEUSDT",
        daily_candles=daily,
        as_of_ms=daily[-1].close_time + 1,
        lookback_days=7,
        min_volume_ratio=1.5,
        min_quote_volume=0.0,
        min_daily_move_pct=0.0,
        max_daily_move_pct=0.25,
        min_daily_range_pct=0.03,
        max_daily_range_pct=0.18,
    )

    assert candidate is None
