from __future__ import annotations

from crypto_signal_bot.paper import CLOSED, PENDING, PaperTrade
from crypto_signal_bot.risk_filter import RiskFilterConfig, filter_signals
from crypto_signal_bot.strategies import ObservationCandidate


def _candidate(symbol: str, direction: str, score: int) -> ObservationCandidate:
    return ObservationCandidate(
        symbol=symbol,
        direction=direction,
        score=score,
        level="强观察",
        current_price=100.0,
        trigger_price=101.0,
        stop_loss=99.0,
        take_profit_1=102.0,
        take_profit_2=103.0,
        target_rr=2.0,
        expires_after_minutes=20,
        expected_hold_hours=2.0,
        reasons=("test",),
    )


def _active_trade(symbol: str, direction: str, score: int, *, t_ms: int = 0) -> PaperTrade:
    return PaperTrade(
        paper_id="T1",
        symbol=symbol,
        direction=direction,
        score=score,
        level="强观察",
        status=PENDING,
        signal_time_ms=t_ms,
        signal_price=100.0,
        confirm_until_ms=t_ms + 10 * 60_000,
        expires_at_ms=t_ms + 20 * 60_000,
        trigger_price=101.0,
        stop_loss=99.0,
        signal_take_profit_1=102.0,
        signal_take_profit_2=103.0,
        target_rr=2.0,
        planned_hold_hours=2.0,
        reasons=(),
    )


def _stopped_trade(symbol: str, direction: str, score: int, *, exit_ms: int) -> PaperTrade:
    return PaperTrade(
        paper_id="T2",
        symbol=symbol,
        direction=direction,
        score=score,
        level="强观察",
        status=CLOSED,
        signal_time_ms=0,
        signal_price=100.0,
        confirm_until_ms=10 * 60_000,
        expires_at_ms=20 * 60_000,
        trigger_price=101.0,
        stop_loss=99.0,
        signal_take_profit_1=102.0,
        signal_take_profit_2=103.0,
        target_rr=2.0,
        planned_hold_hours=2.0,
        reasons=(),
        entry_time_ms=0,
        entry_price=100.0,
        exit_time_ms=exit_ms,
        exit_price=99.0,
        outcome="止损",
    )


def test_altcoin_disabled_filters_non_allowlist_symbols() -> None:
    accepted, skipped = filter_signals(
        [
            _candidate("BTCUSDT", "做多观察", 95),
            _candidate("DOGEUSDT", "做多观察", 97),
        ],
        [],
        now_time_ms=0,
        config=RiskFilterConfig(altcoin_enabled=False, max_open_trades=10, max_open_trades_per_side=10),
    )
    assert [c.symbol for c in accepted] == ["BTCUSDT"]
    assert skipped and skipped[0].symbol == "DOGEUSDT"
    assert skipped[0].skip_reason == "altcoin_disabled"


def test_correlation_filter_skips_eth_when_btc_same_side_active() -> None:
    trades = [_active_trade("BTCUSDT", "做多观察", 94)]
    accepted, skipped = filter_signals(
        [_candidate("ETHUSDT", "做多观察", 95)],
        trades,
        now_time_ms=0,
        config=RiskFilterConfig(
            altcoin_enabled=False,
            max_open_trades=10,
            max_open_trades_per_side=10,
            side_cooldown_minutes=0,
        ),
    )
    assert accepted == []
    assert skipped and skipped[0].skip_reason.startswith("correlated_with_BTCUSDT")


def test_symbol_stop_cooldown_blocks_reentry_same_side() -> None:
    trades = [_stopped_trade("BTCUSDT", "做多观察", 90, exit_ms=0)]
    accepted, skipped = filter_signals(
        [_candidate("BTCUSDT", "做多观察", 99)],
        trades,
        now_time_ms=30 * 60_000,
        config=RiskFilterConfig(
            altcoin_enabled=False,
            max_open_trades=10,
            max_open_trades_per_side=10,
            side_cooldown_minutes=0,
            symbol_cooldown_minutes=60,
        ),
    )
    assert accepted == []
    assert skipped and skipped[0].skip_reason.startswith("symbol_cooldown_after_stop")

