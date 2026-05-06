from __future__ import annotations

from crypto_signal_bot.indicators import (
    compute_moving_averages,
    ma25,
    ma7,
    ma99,
    volume_ma,
)
from crypto_signal_bot.models import Candle


def _c(close: float, vol: float = 1.0, t: int = 0) -> Candle:
    return Candle(t, close, close, close, close, vol, t + 1, True)


def test_sma_ma7_ma25_ma99() -> None:
    candles = [_c(float(i)) for i in range(1, 100)]
    assert ma7(candles) == sum(range(93, 100)) / 7
    assert ma25(candles) == sum(range(75, 100)) / 25
    assert ma99(candles) == sum(range(1, 100)) / 99


def test_volume_ma_default_period() -> None:
    candles = [_c(10.0, vol=float(i)) for i in range(1, 25)]
    expected = sum(range(5, 25)) / 20
    assert volume_ma(candles) == expected


def test_compute_moving_averages_insufficient_history() -> None:
    candles = [_c(1.0), _c(2.0)]
    m = compute_moving_averages(candles)
    assert m.ma7 is None
    assert m.ma25 is None
    assert m.ma99 is None
    assert m.volume_ma is None
