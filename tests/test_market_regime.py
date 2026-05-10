from __future__ import annotations

from crypto_signal_bot.models import Candle
from crypto_signal_bot.strategies.market_regime import detect_market_regime


def _bar(t: int, o: float, h: float, low: float, c: float) -> Candle:
    return Candle(open_time=t, open=o, high=h, low=low, close=c, volume=1000.0, close_time=t + 1, is_closed=True)


def test_no_trade_zone_when_price_in_middle_of_recent_range() -> None:
    # Trend candles: mostly flat -> regime likely range, but no_trade_zone should still trigger.
    trend: list[Candle] = []
    t = 0
    price = 100.0
    for _ in range(140):
        trend.append(_bar(t, price, price + 1.0, price - 1.0, price))
        t += 60_000

    # Entry candles: oscillate within a range; last close near the mid (40%-60%).
    entry: list[Candle] = []
    t = 1_000_000
    for i in range(200):
        close = 95.0 if i % 2 == 0 else 105.0
        if i == 199:
            close = 100.0
        entry.append(_bar(t, close, close + 5.0, close - 5.0, close))
        t += 60_000

    result = detect_market_regime(trend_candles=trend, entry_candles=entry)

    assert result.no_trade_zone is True


def test_trend_regime_allows_trading_context() -> None:
    trend: list[Candle] = []
    entry: list[Candle] = []
    t = 0

    price = 100.0
    for i in range(160):
        price += 0.35
        trend.append(_bar(t, price - 0.2, price + 0.8, price - 0.9, price))
        t += 60_000

    price = 150.0
    for i in range(220):
        price += 0.12
        # Make the last price sit near the upper side of the recent range (avoid mid-zone).
        entry.append(_bar(t, price - 0.1, price + 0.6, price - 0.6, price))
        t += 60_000

    result = detect_market_regime(trend_candles=trend, entry_candles=entry)

    assert result.regime == "trend"
    assert result.no_trade_zone is False
    assert result.confidence >= 70


def test_missing_data_returns_unknown_with_reason() -> None:
    candles = [_bar(i, 1.0, 1.0, 1.0, 1.0) for i in range(10)]

    result = detect_market_regime(trend_candles=candles, entry_candles=candles)

    assert result.regime == "unknown"
    assert result.confidence == 0
    assert any("数据不足" in r or "数据缺失" in r for r in result.reasons)

