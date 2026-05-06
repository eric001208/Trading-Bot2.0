from __future__ import annotations

from crypto_signal_bot.market.binance_kline_stream import build_combined_kline_url, parse_kline_ws_payload


def test_build_combined_kline_url() -> None:
    u = build_combined_kline_url(
        "wss://fstream.binance.com",
        ["BTCUSDT", "ethusdt"],
        ["15m", "1h"],
    )
    assert u.startswith("wss://fstream.binance.com/stream?streams=")
    assert "btcusdt@kline_15m" in u
    assert "btcusdt@kline_1h" in u
    assert "ethusdt@kline_15m" in u
    assert "ethusdt@kline_1h" in u


def test_parse_combined_wrapper() -> None:
    payload = {
        "stream": "btcusdt@kline_15m",
        "data": {
            "e": "kline",
            "E": 1,
            "s": "BTCUSDT",
            "k": {
                "t": 1000,
                "T": 1999,
                "s": "BTCUSDT",
                "i": "15m",
                "o": "1",
                "h": "2",
                "l": "0.5",
                "c": "1.5",
                "v": "10",
                "x": False,
            },
        },
    }
    out = parse_kline_ws_payload(payload)
    assert out is not None
    sym, interval, c = out
    assert sym == "BTCUSDT"
    assert interval == "15m"
    assert c.open_time == 1000
    assert c.close_time == 1999
    assert c.open == 1.0 and c.high == 2.0 and c.low == 0.5 and c.close == 1.5
    assert c.volume == 10.0
    assert c.is_closed is False


def test_parse_single_kline_event() -> None:
    payload = {
        "e": "kline",
        "s": "ETHUSDT",
        "k": {
            "t": 5,
            "T": 6,
            "i": "1h",
            "o": "10",
            "h": "11",
            "l": "9",
            "c": "10.5",
            "v": "3",
            "x": True,
        },
    }
    out = parse_kline_ws_payload(payload)
    assert out is not None
    assert out[0] == "ETHUSDT"
    assert out[1] == "1h"
    assert out[2].is_closed is True


def test_parse_rejects_non_kline() -> None:
    assert parse_kline_ws_payload({"e": "trade"}) is None
    assert parse_kline_ws_payload({}) is None
