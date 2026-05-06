from __future__ import annotations

from crypto_signal_bot.models import Candle
from crypto_signal_bot.strategies.pullback_long import evaluate_pullback_long


def _bar(
    t: int,
    o: float,
    h: float,
    low: float,
    c: float,
    v: float,
) -> Candle:
    return Candle(open_time=t, open=o, high=h, low=low, close=c, volume=v, close_time=t + 1, is_closed=True)


def test_insufficient_history() -> None:
    candles = [_bar(i, 1, 1, 1, 1, 1) for i in range(10)]
    r = evaluate_pullback_long(candles)
    assert r.score == 0
    assert r.direction == "long"
    assert "insufficient" in r.detail


def test_downtrend_low_score() -> None:
    candles: list[Candle] = []
    t = 1_000_000
    price = 200.0
    for _ in range(130):
        price -= 0.8
        c = price
        candles.append(_bar(t, c + 0.2, c + 0.3, c - 0.2, c, 500.0))
        t += 60_000
    r = evaluate_pullback_long(candles)
    assert r.score < 45
    assert "no_uptrend" in r.detail or r.score < 30


def test_pullback_reclaim_high_score() -> None:
    """
    Synthetic uptrend, explicit dip into MA zone in recent window, strong reclaim + volume.
    """
    candles: list[Candle] = []
    t = 2_000_000
    base = 10_000.0

    for i in range(115):
        c = base + i * 12.0
        candles.append(_bar(t, c - 2, c + 4, c - 5, c, 800.0 + i))
        t += 60_000

    for j in range(18):
        c = candles[-1].close - 45.0 - j * 2.0
        low = c - 180.0
        candles.append(_bar(t, c + 10, c + 12, low, c, 650.0))
        t += 60_000

    prev_close = candles[-1].close
    reclaim = prev_close + 950.0
    candles.append(
        _bar(
            t,
            reclaim - 60,
            reclaim + 40,
            reclaim - 70,
            reclaim,
            6000.0,
        )
    )

    r = evaluate_pullback_long(candles)
    assert r.name == "pullback_long"
    assert r.direction == "long"
    assert r.score >= 70, f"expected strong pattern, got score={r.score} detail={r.detail}"
