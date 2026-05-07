from __future__ import annotations

from crypto_signal_bot.models import Candle
from crypto_signal_bot.paper import (
    CLOSED,
    EXPIRED,
    OPEN,
    PENDING,
    load_paper_trades,
    paper_summary_zh,
    record_signal_candidates,
    update_paper_trade_with_candles,
)
from crypto_signal_bot.strategies import ObservationCandidate


def _candidate(direction: str = "做多观察") -> ObservationCandidate:
    return ObservationCandidate(
        symbol="BTCUSDT",
        direction=direction,
        score=95,
        level="强观察",
        current_price=100.0,
        trigger_price=100.5,
        stop_loss=99.0,
        take_profit_1=103.5,
        take_profit_2=105.0,
        target_rr=2.0,
        expires_after_minutes=20,
        expected_hold_hours=2.0,
        reasons=("测试信号",),
    )


def _candle(minutes: int, open_: float, high: float, low: float, close: float) -> Candle:
    open_time = minutes * 60_000
    return Candle(open_time, open_, high, low, close, 1.0, open_time + 5 * 60_000 - 1, True)


def test_record_signal_candidate_creates_pending_trade_and_dedupes(tmp_path) -> None:
    path = tmp_path / "paper.json"

    recorded = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)
    duplicate = record_signal_candidates([_candidate()], path, signal_time_ms=5 * 60_000, confirm_minutes=10)
    trades = load_paper_trades(path)

    assert len(recorded) == 1
    assert duplicate == []
    assert len(trades) == 1
    assert trades[0].status == PENDING
    assert trades[0].confirm_until_ms == 10 * 60_000
    assert trades[0].expires_at_ms == 20 * 60_000


def test_immediate_entry_mode_opens_trade_at_signal_price(tmp_path) -> None:
    path = tmp_path / "paper.json"
    recorded = record_signal_candidates(
        [_candidate()],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
    )
    trade = recorded[0]

    assert trade.status == OPEN
    assert trade.entry_time_ms == 0
    assert trade.entry_price == 100.0


def test_pending_trade_opens_when_trigger_is_confirmed(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    candles = [_candle(1, 100.0, 101.3, 100.0, 101.0)]

    updated = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=6 * 60_000,
        fee_rate=0.0,
        time_stop_minutes=0,
    )

    assert updated.status == OPEN
    assert updated.entry_price == 101.0
    assert updated.entry_time_ms == candles[0].close_time
    assert updated.tp1_price > updated.entry_price
    assert updated.final_target_price > updated.tp1_price


def test_pending_trade_expires_without_confirmation(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]

    updated = update_paper_trade_with_candles(
        trade,
        [],
        now_time_ms=21 * 60_000,
        fee_rate=0.0,
        time_stop_minutes=0,
    )

    assert updated.status == EXPIRED
    assert updated.outcome == "未触发过期"


def test_open_trade_closes_at_stop_after_confirmation(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    candles = [
        _candle(1, 100.0, 101.3, 100.0, 101.0),
        _candle(6, 101.0, 101.1, 98.8, 99.1),
    ]

    updated = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=11 * 60_000,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
    )

    assert updated.status == CLOSED
    assert updated.outcome == "止损"
    assert updated.exit_price == 99.0
    assert updated.pnl_pct < 0


def test_paper_summary_reports_closed_results(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    closed = update_paper_trade_with_candles(
        trade,
        [
            _candle(1, 100.0, 101.3, 100.0, 101.0),
            _candle(6, 101.0, 101.1, 98.8, 99.1),
        ],
        now_time_ms=11 * 60_000,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
    )

    summary = paper_summary_zh([closed])

    assert "虚拟盘记录摘要" in summary
    assert "已平仓：1 笔" in summary
