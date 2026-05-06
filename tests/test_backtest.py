from __future__ import annotations

from crypto_signal_bot.backtest import BacktestSummary, BacktestTrade, _simulate_exit
from crypto_signal_bot.models import Candle


def test_backtest_summary_metrics() -> None:
    trades = (
        BacktestTrade("BTCUSDT", "做多观察", 80, 1, 2, 100.0, 102.0, 99.0, 102.0, "止盈", 0.02),
        BacktestTrade("BTCUSDT", "做多观察", 75, 3, 4, 100.0, 99.0, 99.0, 102.0, "止损", -0.01),
        BacktestTrade("BTCUSDT", "做空观察", 70, 5, 6, 100.0, 99.5, 101.0, 99.0, "到期平仓", 0.005),
    )

    summary = BacktestSummary(
        symbol="BTCUSDT",
        days=7,
        score_threshold=70,
        hold_hours=2.0,
        target_rr=2.0,
        trades=trades,
    )

    assert summary.total == 3
    assert summary.wins == 2
    assert summary.losses == 1
    assert summary.timeouts == 1
    assert round(summary.win_rate, 4) == 0.6667
    assert round(summary.total_return_pct, 4) == 1.5
    assert round(summary.avg_return_pct, 4) == 0.5
    assert round(summary.profit_factor, 2) == 2.5
    assert "BTCUSDT 回测结果" in summary.report_zh()


def test_partial_take_profit_moves_remaining_to_breakeven() -> None:
    entry = Candle(1, 100.0, 100.0, 100.0, 100.0, 1.0, 2, True)
    future = [
        Candle(3, 100.0, 101.2, 100.0, 101.0, 1.0, 4, True),
        Candle(5, 101.0, 101.1, 100.0, 100.0, 1.0, 6, True),
    ]

    trade = _simulate_exit(
        symbol="BTCUSDT",
        direction="做多观察",
        score=90,
        entry_candle=entry,
        future_candles=future,
        stop_loss=99.0,
        take_profit=102.0,
        fee_rate=0.0,
    )

    assert trade.tp1_hit is True
    assert trade.outcome == "半仓止盈后保本"
    assert round(trade.pnl_pct, 4) == 0.005


def test_trailing_stop_locks_profit_after_three_percent_move() -> None:
    entry = Candle(1, 100.0, 100.0, 100.0, 100.0, 1.0, 2, True)
    future = [
        Candle(3, 100.0, 103.5, 100.2, 103.0, 1.0, 4, True),
        Candle(5, 103.0, 103.1, 101.4, 101.5, 1.0, 6, True),
    ]

    trade = _simulate_exit(
        symbol="BTCUSDT",
        direction="做多观察",
        score=90,
        entry_candle=entry,
        future_candles=future,
        stop_loss=98.0,
        take_profit=104.0,
        fee_rate=0.0,
        trailing_trigger_pct=0.03,
        trailing_lock_pct=0.015,
    )

    assert trade.trailing_stop_activated is True
    assert trade.outcome == "移动止损"
    assert round(trade.trailing_stop_price, 4) == 101.5
    assert round(trade.pnl_pct, 4) == 0.0175


def test_funding_cost_is_deducted_by_actual_hold_time() -> None:
    entry = Candle(1, 100.0, 100.0, 100.0, 100.0, 1.0, 0, True)
    future = [
        Candle(3, 100.0, 100.1, 99.9, 100.0, 1.0, 8 * 60 * 60_000, True),
    ]

    trade = _simulate_exit(
        symbol="BTCUSDT",
        direction="做多观察",
        score=90,
        entry_candle=entry,
        future_candles=future,
        stop_loss=90.0,
        take_profit=120.0,
        fee_rate=0.0,
        funding_rate_8h=0.0001,
        r_trailing_enabled=False,
    )

    assert round(trade.funding_cost_pct, 6) == 0.0001
    assert round(trade.pnl_pct, 6) == -0.0001


def test_no_progress_time_stop_exits_early() -> None:
    entry = Candle(1, 100.0, 100.0, 100.0, 100.0, 1.0, 0, True)
    future = [
        Candle(3, 100.0, 100.2, 99.8, 100.1, 1.0, 45 * 60_000, True),
        Candle(4, 100.1, 105.0, 100.1, 104.0, 1.0, 60 * 60_000, True),
    ]

    trade = _simulate_exit(
        symbol="BTCUSDT",
        direction="做多观察",
        score=90,
        entry_candle=entry,
        future_candles=future,
        stop_loss=99.0,
        take_profit=102.0,
        fee_rate=0.0,
        time_stop_minutes=45,
        min_progress_r=0.35,
        r_trailing_enabled=False,
    )

    assert trade.outcome == "时间止损"
    assert trade.exit_time_ms == 45 * 60_000
