from __future__ import annotations

from crypto_signal_bot.models import Candle
from crypto_signal_bot.strategies.observation import (
    ObservationCandidate,
    WeeklyTrend,
    _apply_long_context_filter,
    _position_context,
    classify_weekly_trend,
    estimate_hold_time_plan,
    evaluate_observation_signal,
)


def _bar(t: int, o: float, h: float, low: float, c: float, v: float) -> Candle:
    return Candle(open_time=t, open=o, high=h, low=low, close=c, volume=v, close_time=t + 1, is_closed=True)


def test_observation_long_signal_has_prices_and_reasons() -> None:
    trend: list[Candle] = []
    entry: list[Candle] = []
    t = 1_000_000

    price = 100.0
    for i in range(180):
        price += 0.15
        trend.append(_bar(t, price - 0.05, price + 0.2, price - 0.2, price, 1000 + i))
        t += 60_000

    price = 120.0
    for i in range(165):
        price += 0.03
        entry.append(_bar(t, price - 0.02, price + 0.1, price - 0.1, price, 900 + i))
        t += 60_000

    for _ in range(14):
        price -= 0.12
        entry.append(_bar(t, price + 0.03, price + 0.08, price - 0.25, price, 800))
        t += 60_000

    price += 0.7
    entry.append(_bar(t, price - 0.3, price + 0.2, price - 0.35, price, 2200))

    signal = evaluate_observation_signal(symbol="btcusdt", trend_candles=trend, entry_candles=entry)

    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "做多观察"
    assert signal.score >= 70
    assert signal.stop_loss < signal.reference_entry < signal.take_profit_1 < signal.take_profit_2
    assert signal.reasons


def test_observation_insufficient_history() -> None:
    candles = [_bar(i, 1, 1, 1, 1, 1) for i in range(20)]

    signal = evaluate_observation_signal(symbol="ETHUSDT", trend_candles=candles, entry_candles=candles)

    assert signal.direction == "暂无信号"
    assert signal.score == 0
    assert "历史K线不足" in signal.reasons[0]


def test_weekly_trend_filters_countertrend_side() -> None:
    candles: list[Candle] = []
    price = 100.0
    for i in range(200):
        price += 0.08
        candles.append(_bar(i, price - 0.02, price + 0.1, price - 0.1, price, 1000.0))

    trend = classify_weekly_trend(candles)

    assert trend.direction == "偏多"
    assert trend.allows_long is True
    assert trend.allows_short is False


def test_weekly_trend_marks_sideways_market() -> None:
    candles: list[Candle] = []
    price = 100.0
    for i in range(360):
        price += 0.03 if i % 2 == 0 else -0.03
        candles.append(_bar(i, price - 0.02, price + 0.1, price - 0.1, price, 1000.0))

    trend = classify_weekly_trend(candles)

    assert trend.direction == "震荡"
    assert trend.allows_long is True
    assert trend.allows_short is True


def test_sideways_context_filters_unless_score_is_exceptional() -> None:
    candidate = ObservationCandidate(
        symbol="BTCUSDT",
        direction="做多观察",
        score=94,
        level="强观察",
        current_price=100.0,
        trigger_price=101.0,
        stop_loss=99.0,
        take_profit_1=103.0,
        take_profit_2=104.0,
        target_rr=2.0,
        expires_after_minutes=20,
        expected_hold_hours=2.0,
        reasons=("测试候选",),
    )
    sideways = WeeklyTrend("震荡", True, True, "测试横盘")

    filtered = _apply_long_context_filter(candidate, sideways, label="测试趋势")
    exceptional = _apply_long_context_filter(
        ObservationCandidate(**{**candidate.__dict__, "score": 100}),
        sideways,
        label="测试趋势",
    )

    assert filtered.level == "暂不关注"
    assert filtered.score == 69
    assert exceptional.score == 95
    assert exceptional.level == "强观察"


def test_dynamic_hold_extends_strong_early_trend() -> None:
    trend: list[Candle] = []
    entry: list[Candle] = []
    price = 100.0
    for i in range(110):
        trend.append(_bar(i, price - 0.1, price + 1.0, price - 1.0, price, 1000.0))

    for i in range(24):
        price += 0.25 if i % 2 == 0 else -0.16
        trend.append(_bar(200 + i, price - 0.15, price + 1.1, price - 1.1, price, 1400.0))

    entry_price = price
    for i in range(140):
        entry_price += 0.03
        entry.append(_bar(500 + i, entry_price - 0.05, entry_price + 0.4, entry_price - 0.4, entry_price, 1500.0))

    plan = estimate_hold_time_plan(
        direction="做多观察",
        trend_candles=trend,
        entry_candles=entry,
        base_hold_hours=2.0,
        min_hold_hours=1.0,
        max_hold_hours=4.0,
    )

    assert plan.stage == "趋势初段"
    assert plan.hold_hours > 2.0


def test_position_context_penalizes_near_resistance() -> None:
    candles: list[Candle] = []
    price = 100.0
    for i in range(120):
        price += 0.01
        high = 102.0 if i > 80 else price + 0.2
        candles.append(_bar(i, price - 0.1, high, price - 0.2, price, 1000.0))

    context = _position_context(
        direction="做多观察",
        trend_candles=candles,
        trigger_price=101.9,
        stop_loss=101.0,
    )

    assert context.score_adjustment < 0
    assert "4h上方压力" in context.detail


def test_candidate_telegram_message_contains_manual_checklist() -> None:
    candidate = ObservationCandidate(
        symbol="BTCUSDT",
        direction="做多观察",
        score=96,
        level="强观察",
        current_price=100.0,
        trigger_price=101.0,
        stop_loss=99.0,
        take_profit_1=103.0,
        take_profit_2=104.0,
        target_rr=2.0,
        expires_after_minutes=20,
        expected_hold_hours=2.0,
        reasons=("测试候选",),
    )

    message = candidate.to_telegram_message()

    assert "人工确认清单" in message
    assert "触发后二次确认" in message
    assert "A级" in message
