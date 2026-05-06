from __future__ import annotations

from crypto_signal_bot.candle_store import CandleStore
from crypto_signal_bot.models import Candle, Kline


def _k(t: int, c: float, v: float = 1.0) -> Kline:
    return Kline(open_time=t, open=c, high=c + 1, low=c - 1, close=c, volume=v, close_time=t + 1)


def test_init_and_snapshot() -> None:
    store = CandleStore(max_len=100)
    store.init_from_klines("BTCUSDT", "15m", [_k(1, 10), _k(2, 20)])
    snap = store.snapshot("btcusdt", "15m")
    assert len(snap) == 2
    assert snap[-1].close == 20
    assert snap[-1].is_closed is True


def test_apply_update_same_bar_then_new_bar() -> None:
    store = CandleStore(max_len=10)
    store.init_from_klines("ETHUSDT", "1h", [_k(100, 50)])
    store.apply_update(
        "ETHUSDT",
        "1h",
        Candle(100, 50, 55, 49, 54, 10, 199, is_closed=False),
    )
    s1 = store.snapshot("ethusdt", "1h")
    assert len(s1) == 1
    assert s1[0].close == 54
    assert s1[0].is_closed is False

    store.apply_update(
        "ETHUSDT",
        "1h",
        Candle(100, 50, 55, 49, 53, 12, 199, is_closed=True),
    )
    s2 = store.snapshot("ethusdt", "1h")
    assert s2[0].close == 53
    assert s2[0].is_closed is True

    store.apply_update(
        "ETHUSDT",
        "1h",
        Candle(200, 53, 60, 52, 58, 5, 299, is_closed=False),
    )
    s3 = store.snapshot("ethusdt", "1h")
    assert len(s3) == 2
    assert s3[0].is_closed is True
    assert s3[1].open_time == 200
    assert s3[1].close == 58


def test_max_len_truncates() -> None:
    store = CandleStore(max_len=3)
    store.init_from_klines("X", "15m", [_k(i, float(i)) for i in range(5)])
    snap = store.snapshot("X", "15m")
    assert len(snap) == 3
    assert snap[0].open_time == 2
