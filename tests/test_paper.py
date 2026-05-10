from __future__ import annotations

from crypto_signal_bot.models import Candle
from crypto_signal_bot.paper import (
    BREAKOUT_CONFIRMED,
    CLOSED,
    CANCELLED,
    EXPIRED,
    OPEN,
    PENDING,
    SKIPPED,
    TRIGGERED,
    WAIT_PULLBACK,
    WAITING_HOLD_CONFIRMATION,
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


def _candidate_short() -> ObservationCandidate:
    return ObservationCandidate(
        symbol="BTCUSDT",
        direction="做空观察",
        score=95,
        level="强观察",
        current_price=100.0,
        trigger_price=99.5,
        stop_loss=101.0,
        take_profit_1=98.5,
        take_profit_2=97.0,
        target_rr=2.0,
        expires_after_minutes=20,
        expected_hold_hours=2.0,
        reasons=("测试空头信号",),
    )


def _candle(minutes: int, open_: float, high: float, low: float, close: float) -> Candle:
    open_time = minutes * 60_000
    return Candle(open_time, open_, high, low, close, 1.0, open_time + 5 * 60_000 - 1, True)


def _bar(t_ms: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(t_ms, open_, high, low, close, 1.0, t_ms + 5 * 60_000, True)


def _event_statuses(trade) -> list[str]:
    events = getattr(trade, "status_events", ()) or ()
    return [str(e.get("status", "")) for e in events]


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


def test_breakout_close_and_hold_long_waits_then_triggers(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )[0]
    breakout = _bar(0, 100.0, 100.9, 99.9, 100.6)

    waiting = update_paper_trade_with_candles(
        trade,
        [breakout],
        now_time_ms=breakout.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )
    assert waiting.status == WAITING_HOLD_CONFIRMATION
    assert _event_statuses(waiting)[:3] == [PENDING, BREAKOUT_CONFIRMED, WAITING_HOLD_CONFIRMATION]

    hold = _bar(5 * 60_000, 100.6, 100.9, 100.4, 100.7)
    triggered = update_paper_trade_with_candles(
        waiting,
        [hold],
        now_time_ms=hold.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )
    assert triggered.status == TRIGGERED
    assert TRIGGERED in _event_statuses(triggered)


def test_breakout_close_and_hold_long_cancels_if_close_back_below_trigger(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )[0]
    breakout = _bar(0, 100.0, 100.9, 99.9, 100.6)
    hold_fail = _bar(5 * 60_000, 100.6, 100.7, 100.2, 100.4)

    updated = update_paper_trade_with_candles(
        trade,
        [breakout, hold_fail],
        now_time_ms=hold_fail.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )
    assert updated.status == CANCELLED
    assert updated.entry_time_ms is None


def test_breakout_close_and_hold_short_waits_then_triggers(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate_short()],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )[0]
    breakout = _bar(0, 100.0, 100.2, 99.2, 99.4)

    waiting = update_paper_trade_with_candles(
        trade,
        [breakout],
        now_time_ms=breakout.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )
    assert waiting.status == WAITING_HOLD_CONFIRMATION

    hold = _bar(5 * 60_000, 99.4, 99.6, 99.0, 99.3)
    triggered = update_paper_trade_with_candles(
        waiting,
        [hold],
        now_time_ms=hold.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_hold",
        hold_confirmation_candles=1,
    )
    assert triggered.status == TRIGGERED
    assert triggered.entry_price == hold.close


def test_breakout_close_and_pullback_long_waits_pullback_then_triggers(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )[0]
    breakout = _bar(0, 100.0, 100.9, 99.9, 100.6)

    waiting = update_paper_trade_with_candles(
        trade,
        [breakout],
        now_time_ms=breakout.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )
    assert waiting.status == WAIT_PULLBACK
    assert waiting.wait_pullback is True

    pullback = _bar(5 * 60_000, 100.6, 100.8, 100.35, 100.55)
    triggered = update_paper_trade_with_candles(
        waiting,
        [pullback],
        now_time_ms=pullback.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )
    assert triggered.status == TRIGGERED
    assert triggered.entry_price == pullback.close


def test_breakout_close_and_pullback_short_waits_pullback_then_triggers(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate_short()],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )[0]
    breakout = _bar(0, 100.0, 100.2, 99.2, 99.4)

    waiting = update_paper_trade_with_candles(
        trade,
        [breakout],
        now_time_ms=breakout.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )
    assert waiting.status == WAIT_PULLBACK
    assert waiting.wait_pullback is True

    pullback = _bar(5 * 60_000, 99.4, 99.6, 99.2, 99.45)
    triggered = update_paper_trade_with_candles(
        waiting,
        [pullback],
        now_time_ms=pullback.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_and_pullback",
        pullback_tolerance_r=0.25,
        pullback_expire_minutes=30,
    )
    assert triggered.status == TRIGGERED
    assert triggered.entry_price == pullback.close


def test_legacy_mode_triggers_on_touch_while_close_only_does_not(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate("做多观察")], path, signal_time_ms=0, confirm_minutes=10)[0]
    # High touches trigger (100.5) but close is back below.
    touch_only = _bar(0, 100.0, 100.7, 99.9, 100.4)

    legacy = update_paper_trade_with_candles(
        trade,
        [touch_only],
        now_time_ms=touch_only.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="legacy",
    )
    assert legacy.status == TRIGGERED

    close_only = update_paper_trade_with_candles(
        trade,
        [touch_only],
        now_time_ms=touch_only.close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_only",
    )
    assert close_only.status == PENDING


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


def test_r_multiple_tp_prices_are_computed_from_entry_and_stop(tmp_path) -> None:
    path = tmp_path / "paper.json"
    long_trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
        tp1_r=1.0,
        tp2_r=2.0,
        final_tp_r=3.0,
    )[0]
    assert long_trade.entry_price == 100.0
    assert long_trade.stop_loss == 99.0
    assert long_trade.tp1_price == 101.0
    assert long_trade.tp2_price == 102.0
    assert long_trade.final_target_price == 103.0

    short_trade = record_signal_candidates(
        [_candidate_short()],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
        tp1_r=1.0,
        tp2_r=2.0,
        final_tp_r=3.0,
    )[0]
    assert short_trade.entry_price == 100.0
    assert short_trade.stop_loss == 101.0
    assert short_trade.tp1_price == 99.0
    assert short_trade.tp2_price == 98.0
    assert short_trade.final_target_price == 97.0


def test_move_stop_to_breakeven_after_tp1_can_be_disabled(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
        tp1_r=1.0,
        tp2_r=2.0,
        final_tp_r=3.0,
        partial_close_pct_at_tp1=0.5,
    )[0]
    candles = [
        _bar(0, 100.0, 101.2, 100.0, 101.0),  # hit TP1
        _bar(5 * 60_000, 101.0, 101.0, 99.0, 99.5),  # flush down to stop zone
    ]

    moved = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        move_stop_to_breakeven_after_tp1=True,
    )
    assert moved.status == CLOSED
    assert moved.tp1_hit is True
    assert moved.moved_stop_to_breakeven is True
    assert round(moved.pnl_pct, 6) == 0.005

    not_moved = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        move_stop_to_breakeven_after_tp1=False,
    )
    assert not_moved.status == CLOSED
    assert not_moved.tp1_hit is True
    assert not_moved.moved_stop_to_breakeven is False
    assert round(not_moved.pnl_pct, 6) == 0.0


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
        breakout_confirmation_mode="close_only",
    )

    assert updated.status == TRIGGERED
    assert updated.trigger_confirmed is True
    assert updated.trigger_time_ms == candles[0].close_time
    assert updated.trigger_fill_price == candles[0].close
    assert updated.entry_price == 101.0
    assert updated.entry_time_ms == candles[0].close_time
    assert updated.tp1_price > updated.entry_price
    assert updated.final_target_price > updated.tp1_price


def test_no_chase_switches_to_wait_pullback_and_enters_on_pullback(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    candles = [
        _bar(0, 100.6, 102.2, 100.5, 102.0),  # trigger confirmed, but too far from trigger -> wait_pullback
        _bar(5 * 60_000, 102.0, 102.1, 100.4, 100.6),  # pullback to trigger area -> enter
    ]

    waited = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_only",
    )
    assert waited.status == WAIT_PULLBACK
    assert waited.entry_time_ms is None
    assert waited.wait_pullback is True
    assert waited.no_chase_passed is False
    assert waited.trigger_time_ms == candles[0].close_time
    assert waited.trigger_fill_price == candles[0].close
    assert waited.entry_slippage_r is not None and waited.entry_slippage_r > 0.9
    assert "entry_slippage_r" in waited.skip_reason

    triggered = update_paper_trade_with_candles(
        waited,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_only",
    )
    assert triggered.status == TRIGGERED
    assert triggered.entry_time_ms == candles[1].close_time
    assert triggered.entry_price == candles[1].close
    # Keep the original trigger info for auditing.
    assert triggered.trigger_time_ms == candles[0].close_time
    assert triggered.trigger_fill_price == candles[0].close


def test_no_chase_can_skip_instead_of_wait_pullback(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    candles = [_bar(0, 100.6, 102.2, 100.5, 102.0)]

    skipped = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        wait_pullback_if_chased=False,
        breakout_confirmation_mode="close_only",
    )
    assert skipped.status == SKIPPED
    assert skipped.entry_time_ms is None
    assert skipped.no_chase_passed is False
    assert skipped.exit_time_ms == candles[0].close_time
    assert skipped.exit_price == candles[0].close
    assert "entry_slippage_r" in skipped.skip_reason


def _disabled_test_no_chase_wait_pullback_when_trigger_candle_atr_multiple_too_high(tmp_path) -> None:
    path = tmp_path / "paper.json"
    candidate = ObservationCandidate(
        symbol="BTCUSDT",
        direction="åšå¤šè§‚å¯˜".replace("å¯˜", "å¯˜"),
        score=95,
        level="å¼ºè§‚å¯˜".replace("å¯˜", "å¯˜"),
        current_price=100.0,
        trigger_price=100.5,
        stop_loss=99.0,
        take_profit_1=103.5,
        take_profit_2=105.0,
        target_rr=2.0,
        expires_after_minutes=120,
        expected_hold_hours=2.0,
        reasons=("atr spike test",),
    )
    trade = record_signal_candidates([candidate], path, signal_time_ms=0, confirm_minutes=10)[0]
    calm = [_bar(i * 5 * 60_000, 100.0, 100.05, 99.95, 100.0) for i in range(15)]
    trigger_spike = _bar(15 * 5 * 60_000, 100.0, 101.25, 99.1, 101.2)
    candles = calm + [trigger_spike]

    waited = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_only",
    )
    assert waited.status == WAIT_PULLBACK
    assert waited.no_chase_passed is False
    assert waited.trigger_candle_atr_multiple is not None
    assert waited.trigger_candle_atr_multiple > 1.2
    assert "trigger_candle_atr_multiple" in waited.skip_reason


def test_no_chase_wait_pullback_when_trigger_candle_atr_multiple_too_high(tmp_path) -> None:
    path = tmp_path / "paper.json"
    candidate = ObservationCandidate(
        symbol="BTCUSDT",
        direction="做多观察",
        score=95,
        level="强观察",
        current_price=100.0,
        trigger_price=100.5,
        stop_loss=99.0,
        take_profit_1=103.5,
        take_profit_2=105.0,
        target_rr=2.0,
        expires_after_minutes=120,
        expected_hold_hours=2.0,
        reasons=("atr spike test",),
    )
    trade = record_signal_candidates([candidate], path, signal_time_ms=0, confirm_minutes=10)[0]

    calm = [_bar(i * 5 * 60_000, 100.0, 100.05, 99.95, 100.0) for i in range(15)]
    trigger_spike = _bar(15 * 5 * 60_000, 100.0, 101.25, 99.1, 101.2)
    candles = calm + [trigger_spike]

    waited = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
        breakout_confirmation_mode="close_only",
    )
    assert waited.status == WAIT_PULLBACK
    assert waited.no_chase_passed is False
    assert waited.trigger_candle_atr_multiple is not None
    assert waited.trigger_candle_atr_multiple > 1.2
    assert "trigger_candle_atr_multiple" in waited.skip_reason


def test_pending_signal_is_cancelled_if_stop_is_hit_before_trigger(tmp_path) -> None:
    path = tmp_path / "paper.json"
    trade = record_signal_candidates([_candidate()], path, signal_time_ms=0, confirm_minutes=10)[0]
    # Price hits stop (99.0) without ever closing above trigger (100.5).
    candles = [
        _candle(1, 100.0, 100.4, 98.8, 99.6),
    ]

    updated = update_paper_trade_with_candles(
        trade,
        candles,
        now_time_ms=6 * 60_000,
        fee_rate=0.0,
        time_stop_minutes=0,
    )

    assert updated.status == CANCELLED
    assert updated.entry_time_ms is None


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
        breakout_confirmation_mode="close_only",
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
        breakout_confirmation_mode="close_only",
    )

    summary = paper_summary_zh([closed])

    assert "虚拟盘记录摘要" in summary
    assert "已平仓：1 笔" in summary


def test_open_trade_tracks_mfe_mae_for_long_and_short(tmp_path) -> None:
    path = tmp_path / "paper.json"

    long_trade = record_signal_candidates(
        [_candidate("做多观察")],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
    )[0]
    long_candles = [
        _bar(0, 100.0, 101.5, 99.7, 101.0),
        _bar(5 * 60_000, 101.0, 102.0, 99.2, 100.5),
        _bar(10 * 60_000, 100.5, 101.0, 99.5, 100.0),
    ]
    updated_long = update_paper_trade_with_candles(
        long_trade,
        long_candles,
        now_time_ms=long_candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
    )
    assert updated_long.max_favorable_price == 102.0
    assert updated_long.max_adverse_price == 99.2
    assert round(updated_long.max_favorable_pnl_pct, 6) == 0.02
    assert round(updated_long.max_adverse_pnl_pct, 6) == -0.008
    assert round(updated_long.max_favorable_r, 6) == 2.0
    assert round(updated_long.max_adverse_r, 6) == -0.8
    assert updated_long.time_to_mfe_minutes == 10.0
    assert updated_long.time_to_mae_minutes == 10.0

    short_trade = record_signal_candidates(
        [_candidate_short()],
        path,
        signal_time_ms=0,
        confirm_minutes=10,
        entry_mode="immediate",
    )[0]
    short_candles = [
        _bar(0, 100.0, 100.8, 99.0, 99.6),
        _bar(5 * 60_000, 99.6, 100.6, 98.5, 99.0),
    ]
    updated_short = update_paper_trade_with_candles(
        short_trade,
        short_candles,
        now_time_ms=short_candles[-1].close_time + 1,
        fee_rate=0.0,
        time_stop_minutes=0,
        r_trailing_enabled=False,
        trailing_trigger_pct=0.0,
    )
    assert updated_short.max_favorable_price == 98.5
    assert updated_short.max_adverse_price == 100.8
    assert round(updated_short.max_favorable_pnl_pct, 6) == 0.015
    assert round(updated_short.max_adverse_pnl_pct, 6) == -0.008
    assert round(updated_short.max_favorable_r, 6) == 1.5
    assert round(updated_short.max_adverse_r, 6) == -0.8
    assert updated_short.time_to_mfe_minutes == 10.0
    assert updated_short.time_to_mae_minutes == 5.0
