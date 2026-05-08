from __future__ import annotations

import csv
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crypto_signal_bot.market import fetch_usdm_klines_range
from crypto_signal_bot.models import Candle
from crypto_signal_bot.strategies import ObservationCandidate, evaluate_observation_candidate

ENTRY_INTERVAL_MS = 15 * 60_000
TREND_INTERVAL_MS = 60 * 60_000
FUNDING_INTERVAL_MS = 8 * 60 * 60_000
DEFAULT_TREND_WINDOW_BARS = 360  # match live scanner (about 15 days of 1h bars)
DEFAULT_ENTRY_WINDOW_BARS = 180  # match live scanner (about 45 hours of 15m bars)


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    direction: str
    score: int
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    outcome: str
    pnl_pct: float
    tp1_price: float = 0.0
    tp1_hit: bool = False
    final_target_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_activated: bool = False
    planned_hold_hours: float = 0.0
    funding_cost_pct: float = 0.0


@dataclass(frozen=True)
class BacktestSummary:
    symbol: str
    days: int
    score_threshold: int
    hold_hours: float
    target_rr: float
    trades: tuple[BacktestTrade, ...]
    funding_rate_8h: float = 0.0

    @property
    def total(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct < 0)

    @property
    def timeouts(self) -> int:
        return sum(1 for t in self.trades if t.outcome in {"到期平仓", "半仓止盈后到期"})

    @property
    def time_stops(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "时间止损")

    @property
    def trailing_activations(self) -> int:
        return sum(1 for t in self.trades if t.trailing_stop_activated)

    @property
    def trailing_exits(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "移动止损")

    @property
    def total_funding_cost_pct(self) -> float:
        return sum(t.funding_cost_pct for t in self.trades) * 100

    @property
    def avg_planned_hold_hours(self) -> float:
        planned = [t.planned_hold_hours for t in self.trades if t.planned_hold_hours > 0]
        if not planned:
            return self.hold_hours
        return sum(planned) / len(planned)

    def direction_total(self, direction: str) -> int:
        return sum(1 for t in self.trades if t.direction == direction)

    def direction_win_rate(self, direction: str) -> float:
        trades = [t for t in self.trades if t.direction == direction]
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.pnl_pct > 0) / len(trades)

    def direction_return_pct(self, direction: str) -> float:
        return sum(t.pnl_pct for t in self.trades if t.direction == direction) * 100

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def total_return_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades) * 100

    @property
    def avg_return_pct(self) -> float:
        return self.total_return_pct / self.total if self.total else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        losses = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    @property
    def max_drawdown_pct(self) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in self.trades:
            equity += trade.pnl_pct
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity - peak)
        return max_drawdown * 100

    def report_zh(self) -> str:
        profit_factor = "无限" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        return (
            f"{self.symbol} 回测结果\n"
            f"回测天数：{self.days}\n"
            f"评分阈值：{self.score_threshold}\n"
            f"基础预计持仓：{self.hold_hours:g} 小时\n"
            f"平均动态持仓：{self.avg_planned_hold_hours:.2f} 小时\n"
            f"目标盈亏比：{self.target_rr:g}:1\n"
            f"资金费率假设：每8小时 {self.funding_rate_8h * 100:.4f}%\n"
            f"资金费率成本：{self.total_funding_cost_pct:.3f}%\n"
            f"交易次数：{self.total}\n"
            f"胜率：{self.win_rate * 100:.2f}%\n"
            f"累计收益率：{self.total_return_pct:.2f}%\n"
            f"平均单笔收益率：{self.avg_return_pct:.3f}%\n"
            f"最大回撤：{self.max_drawdown_pct:.2f}%\n"
            f"盈亏因子：{profit_factor}\n"
            f"盈利笔数：{self.wins}\n"
            f"亏损笔数：{self.losses}\n"
            f"到期平仓笔数：{self.timeouts}\n"
            f"时间止损笔数：{self.time_stops}\n"
            f"移动止损触发：{self.trailing_activations}\n"
            f"移动止损出场：{self.trailing_exits}\n"
            f"做多：{self.direction_total('做多观察')} 笔，胜率 {self.direction_win_rate('做多观察') * 100:.2f}%，收益 {self.direction_return_pct('做多观察'):.2f}%\n"
            f"做空：{self.direction_total('做空观察')} 笔，胜率 {self.direction_win_rate('做空观察') * 100:.2f}%，收益 {self.direction_return_pct('做空观察'):.2f}%"
        )


def _to_candles(klines: Sequence) -> list[Candle]:
    return [Candle.from_kline(k) for k in klines]


def _trend_slice_for_time(trend: Sequence[Candle], close_time_ms: int) -> list[Candle]:
    return [c for c in trend if c.close_time <= close_time_ms]


def _entry_slice_for_time(entry: Sequence[Candle], close_time_ms: int) -> list[Candle]:
    return [c for c in entry if c.close_time <= close_time_ms]


def _pnl_pct(direction: str, entry: float, exit_price: float, fee_rate: float) -> float:
    gross = (exit_price - entry) / entry
    if direction == "做空观察":
        gross = (entry - exit_price) / entry
    return gross - 2 * fee_rate


def _gross_pct(direction: str, entry: float, exit_price: float) -> float:
    gross = (exit_price - entry) / entry
    if direction == "做空观察":
        gross = (entry - exit_price) / entry
    return gross


def _partial_pnl_pct(
    *,
    direction: str,
    entry: float,
    first_exit: float,
    final_exit: float,
    fee_rate: float,
    first_weight: float = 0.5,
) -> float:
    second_weight = 1.0 - first_weight
    gross = first_weight * _gross_pct(direction, entry, first_exit)
    gross += second_weight * _gross_pct(direction, entry, final_exit)
    return gross - 2 * fee_rate


def _funding_cost_pct(entry_time_ms: int, exit_time_ms: int, funding_rate_8h: float) -> float:
    if funding_rate_8h <= 0:
        return 0.0
    duration_ms = max(0, exit_time_ms - entry_time_ms)
    return funding_rate_8h * duration_ms / FUNDING_INTERVAL_MS


def _make_trade(
    *,
    symbol: str,
    direction: str,
    score: int,
    entry_time_ms: int,
    exit_time_ms: int,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    take_profit: float,
    outcome: str,
    pnl_pct: float,
    funding_rate_8h: float,
    tp1_price: float,
    tp1_hit: bool,
    final_target_price: float,
    trailing_stop_price: float,
    trailing_stop_activated: bool,
    planned_hold_hours: float,
) -> BacktestTrade:
    funding_cost = _funding_cost_pct(entry_time_ms, exit_time_ms, funding_rate_8h)
    return BacktestTrade(
        symbol=symbol,
        direction=direction,
        score=score,
        entry_time_ms=entry_time_ms,
        exit_time_ms=exit_time_ms,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        outcome=outcome,
        pnl_pct=pnl_pct - funding_cost,
        tp1_price=tp1_price,
        tp1_hit=tp1_hit,
        final_target_price=final_target_price,
        trailing_stop_price=trailing_stop_price,
        trailing_stop_activated=trailing_stop_activated,
        planned_hold_hours=planned_hold_hours,
        funding_cost_pct=funding_cost,
    )


def _r_trailing_stop_price(
    *,
    direction: str,
    entry: float,
    risk: float,
    best_progress_r: float,
    tp1_hit: bool,
) -> float | None:
    lock_r: float | None = None
    if tp1_hit:
        if best_progress_r >= 2.0:
            lock_r = 0.75
        elif best_progress_r >= 1.5:
            lock_r = 0.25
    elif best_progress_r >= 0.8:
        lock_r = -0.3

    if lock_r is None:
        return None
    if direction == "做多观察":
        return entry + lock_r * risk
    return entry - lock_r * risk


def _simulate_exit(
    *,
    symbol: str,
    direction: str,
    score: int,
    entry_candle: Candle,
    future_candles: Sequence[Candle],
    stop_loss: float,
    take_profit: float,
    fee_rate: float,
    funding_rate_8h: float = 0.0,
    time_stop_minutes: int = 0,
    min_progress_r: float = 0.35,
    r_trailing_enabled: bool = True,
    trailing_trigger_pct: float = 0.03,
    trailing_lock_pct: float = 0.015,
    planned_hold_hours: float = 0.0,
) -> BacktestTrade:
    entry = entry_candle.close
    is_long = direction == "做多观察"
    last = future_candles[-1]
    risk = max(abs(entry - stop_loss), entry * 0.001)
    tp1 = entry + risk if is_long else entry - risk
    tp2 = take_profit
    active_stop = stop_loss
    trailing_stop = 0.0
    trailing_active = False
    tp1_hit = False
    tp1_time = 0
    time_stop_at = entry_candle.close_time + time_stop_minutes * 60_000
    best_progress_r = 0.0

    for candle in future_candles:
        if is_long:
            best_progress_r = max(best_progress_r, (candle.high - entry) / risk)
            if candle.low <= active_stop:
                outcome = "移动止损" if trailing_active else ("半仓止盈后保本" if tp1_hit else "止损")
                pnl = (
                    _partial_pnl_pct(
                        direction=direction,
                        entry=entry,
                        first_exit=tp1,
                        final_exit=active_stop,
                        fee_rate=fee_rate,
                    )
                    if tp1_hit
                    else _pnl_pct(direction, entry, active_stop, fee_rate)
                )
                return _make_trade(
                    symbol=symbol,
                    direction=direction,
                    score=score,
                    entry_time_ms=entry_candle.close_time,
                    exit_time_ms=candle.close_time,
                    entry_price=entry,
                    exit_price=active_stop,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    tp1_hit=tp1_hit,
                    final_target_price=tp2,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    planned_hold_hours=planned_hold_hours,
                )
            if r_trailing_enabled:
                r_stop = _r_trailing_stop_price(
                    direction=direction,
                    entry=entry,
                    risk=risk,
                    best_progress_r=best_progress_r,
                    tp1_hit=tp1_hit,
                )
                if r_stop is not None and r_stop > active_stop:
                    active_stop = r_stop
                    trailing_stop = active_stop
                    if tp1_hit:
                        trailing_active = True
            if trailing_trigger_pct > 0 and not trailing_active and candle.high >= entry * (1 + trailing_trigger_pct):
                trailing_active = True
                trailing_stop = entry * (1 + trailing_lock_pct)
                active_stop = max(active_stop, trailing_stop)
            if not tp1_hit and candle.high >= tp1:
                tp1_hit = True
                tp1_time = candle.close_time
                active_stop = max(active_stop, entry)
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.high >= tp2:
                pnl = _partial_pnl_pct(
                    direction=direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=tp2,
                    fee_rate=fee_rate,
                )
                return _make_trade(
                    symbol=symbol,
                    direction=direction,
                    score=score,
                    entry_time_ms=entry_candle.close_time,
                    exit_time_ms=candle.close_time,
                    entry_price=entry,
                    exit_price=tp2,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    tp1_hit=True,
                    final_target_price=tp2,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    planned_hold_hours=planned_hold_hours,
                )
        else:
            best_progress_r = max(best_progress_r, (entry - candle.low) / risk)
            if candle.high >= active_stop:
                outcome = "移动止损" if trailing_active else ("半仓止盈后保本" if tp1_hit else "止损")
                pnl = (
                    _partial_pnl_pct(
                        direction=direction,
                        entry=entry,
                        first_exit=tp1,
                        final_exit=active_stop,
                        fee_rate=fee_rate,
                    )
                    if tp1_hit
                    else _pnl_pct(direction, entry, active_stop, fee_rate)
                )
                return _make_trade(
                    symbol=symbol,
                    direction=direction,
                    score=score,
                    entry_time_ms=entry_candle.close_time,
                    exit_time_ms=candle.close_time,
                    entry_price=entry,
                    exit_price=active_stop,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    tp1_hit=tp1_hit,
                    final_target_price=tp2,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    planned_hold_hours=planned_hold_hours,
                )
            if r_trailing_enabled:
                r_stop = _r_trailing_stop_price(
                    direction=direction,
                    entry=entry,
                    risk=risk,
                    best_progress_r=best_progress_r,
                    tp1_hit=tp1_hit,
                )
                if r_stop is not None and r_stop < active_stop:
                    active_stop = r_stop
                    trailing_stop = active_stop
                    if tp1_hit:
                        trailing_active = True
            if trailing_trigger_pct > 0 and not trailing_active and candle.low <= entry * (1 - trailing_trigger_pct):
                trailing_active = True
                trailing_stop = entry * (1 - trailing_lock_pct)
                active_stop = min(active_stop, trailing_stop)
            if not tp1_hit and candle.low <= tp1:
                tp1_hit = True
                tp1_time = candle.close_time
                active_stop = min(active_stop, entry)
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.low <= tp2:
                pnl = _partial_pnl_pct(
                    direction=direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=tp2,
                    fee_rate=fee_rate,
                )
                return _make_trade(
                    symbol=symbol,
                    direction=direction,
                    score=score,
                    entry_time_ms=entry_candle.close_time,
                    exit_time_ms=candle.close_time,
                    entry_price=entry,
                    exit_price=tp2,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    tp1_hit=True,
                    final_target_price=tp2,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    planned_hold_hours=planned_hold_hours,
                )
        if time_stop_minutes > 0 and candle.close_time >= time_stop_at and best_progress_r < min_progress_r:
            pnl = (
                _partial_pnl_pct(
                    direction=direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=candle.close,
                    fee_rate=fee_rate,
                )
                if tp1_hit
                else _pnl_pct(direction, entry, candle.close, fee_rate)
            )
            return _make_trade(
                symbol=symbol,
                direction=direction,
                score=score,
                entry_time_ms=entry_candle.close_time,
                exit_time_ms=candle.close_time,
                entry_price=entry,
                exit_price=candle.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                outcome="时间止损",
                pnl_pct=pnl,
                funding_rate_8h=funding_rate_8h,
                tp1_price=tp1,
                tp1_hit=tp1_hit,
                final_target_price=tp2,
                trailing_stop_price=trailing_stop,
                trailing_stop_activated=trailing_active,
                planned_hold_hours=planned_hold_hours,
            )

    pnl = (
        _partial_pnl_pct(
            direction=direction,
            entry=entry,
            first_exit=tp1,
            final_exit=last.close,
            fee_rate=fee_rate,
        )
        if tp1_hit
        else _pnl_pct(direction, entry, last.close, fee_rate)
    )
    return _make_trade(
        symbol=symbol,
        direction=direction,
        score=score,
        entry_time_ms=entry_candle.close_time,
        exit_time_ms=last.close_time,
        entry_price=entry,
        exit_price=last.close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        outcome="半仓止盈后到期" if tp1_hit else "到期平仓",
        pnl_pct=pnl,
        funding_rate_8h=funding_rate_8h,
        tp1_price=tp1,
        tp1_hit=tp1_hit,
        final_target_price=tp2,
        trailing_stop_price=trailing_stop,
        trailing_stop_activated=trailing_active,
        planned_hold_hours=planned_hold_hours,
    )


def _find_confirmation_entry(
    *,
    candidate: ObservationCandidate,
    confirmation_candles: Sequence[Candle],
    candidate_close_time_ms: int,
    confirm_minutes: int,
    expire_minutes: int,
) -> Candle | None:
    confirm_deadline = candidate_close_time_ms + confirm_minutes * 60_000
    expire_deadline = candidate_close_time_ms + expire_minutes * 60_000
    deadline = min(confirm_deadline, expire_deadline)

    for candle in confirmation_candles:
        if candle.close_time <= candidate_close_time_ms:
            continue
        if candle.close_time > deadline:
            break
        if candidate.direction == "做多观察":
            if candle.close >= candidate.trigger_price and candle.close >= candle.open:
                return candle
        elif candidate.direction == "做空观察":
            if candle.close <= candidate.trigger_price and candle.close <= candle.open:
                return candle
    return None


def _candles_between(
    candles: Sequence[Candle],
    *,
    start_after_ms: int,
    end_at_ms: int,
) -> list[Candle]:
    return [c for c in candles if c.close_time > start_after_ms and c.close_time <= end_at_ms]


def _opposite_direction(a: str, b: str) -> bool:
    return {a, b} == {"做多观察", "做空观察"}


def _threshold_for_candidate(
    candidate: ObservationCandidate,
    default: int,
    direction_thresholds: dict[tuple[str, str], int] | None,
) -> int:
    if not direction_thresholds:
        return default
    return direction_thresholds.get((candidate.symbol.strip().upper(), candidate.direction), default)


def _has_conflicting_market_signal(
    *,
    candidate: ObservationCandidate,
    market_trend_candles: Sequence[Candle] | None,
    market_entry_candles: Sequence[Candle] | None,
    candidate_close_time_ms: int,
    hold_hours: float,
    target_rr: float,
    expire_minutes: int,
    conflict_threshold: int,
    conflict_minutes: int,
) -> bool:
    if market_trend_candles is None or market_entry_candles is None:
        return False
    window_start = candidate_close_time_ms - conflict_minutes * 60_000
    for idx, candle in enumerate(market_entry_candles):
        if candle.close_time < window_start or candle.close_time > candidate_close_time_ms:
            continue
        trend_slice = _trend_slice_for_time(market_trend_candles, candle.close_time)
        if len(trend_slice) < 120 or idx < 120:
            continue
        btc_candidate = evaluate_observation_candidate(
            symbol="BTCUSDT",
            trend_candles=trend_slice,
            entry_candles=market_entry_candles[: idx + 1],
            expected_hold_hours=hold_hours,
            target_rr=target_rr,
            expires_after_minutes=expire_minutes,
            hard_volume_filter=False,
            hard_short_trend_filter=False,
        )
        if (
            btc_candidate.score >= conflict_threshold
            and btc_candidate.direction in {"做多观察", "做空观察"}
            and _opposite_direction(candidate.direction, btc_candidate.direction)
        ):
            return True
    return False


def run_observation_backtest_from_candles(
    *,
    symbol: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    confirmation_candles: Sequence[Candle] | None = None,
    market_trend_candles: Sequence[Candle] | None = None,
    market_entry_candles: Sequence[Candle] | None = None,
    score_threshold: int = 70,
    hold_hours: float = 2.0,
    fee_rate: float = 0.0005,
    target_rr: float = 2.0,
    confirm_minutes: int = 10,
    expire_minutes: int = 20,
    funding_rate_8h: float = 0.0,
    time_stop_minutes: int = 45,
    min_progress_r: float = 0.35,
    r_trailing_enabled: bool = True,
    trailing_trigger_pct: float = 0.03,
    trailing_lock_pct: float = 0.015,
    weekly_filter: bool = True,
    dynamic_hold: bool = True,
    min_hold_hours: float = 1.0,
    max_hold_hours: float = 4.0,
    allowed_entry_days: set[str] | None = None,
    allowed_directions: set[str] | None = None,
    direction_thresholds: dict[tuple[str, str], int] | None = None,
    conflict_threshold: int = 90,
    conflict_minutes: int = 60,
    cooldown_bars: int | None = None,
    days: int = 0,
    start_index: int = 120,
    trend_window_bars: int = DEFAULT_TREND_WINDOW_BARS,
    entry_window_bars: int = DEFAULT_ENTRY_WINDOW_BARS,
) -> BacktestSummary:
    max_configured_hold_hours = max_hold_hours if dynamic_hold else hold_hours
    max_hold_bars = max(1, int(round(max_configured_hold_hours * 60 / 15)))
    cooldown = max_hold_bars if cooldown_bars is None else max(0, cooldown_bars)
    trades: list[BacktestTrade] = []
    entry = list(entry_candles)
    trend = list(trend_candles)
    market_trend = list(market_trend_candles) if market_trend_candles is not None else None
    market_entry = list(market_entry_candles) if market_entry_candles is not None else None

    # Precompute keys for fast bisection. This keeps long backtests (e.g. 2025->now) practical.
    trend_close_times = [c.close_time for c in trend]
    market_trend_close_times = [c.close_time for c in market_trend] if market_trend is not None else None
    market_entry_close_times = [c.close_time for c in market_entry] if market_entry is not None else None

    i = max(120, start_index)
    last_entry_index = len(entry) - max_hold_bars - 1
    lower_timeframe = list(confirmation_candles or entry)
    lower_close_times = [c.close_time for c in lower_timeframe]

    def _between(start_after_ms: int, end_at_ms: int) -> list[Candle]:
        if end_at_ms <= start_after_ms:
            return []
        start_idx = bisect_right(lower_close_times, start_after_ms)
        end_idx = bisect_right(lower_close_times, end_at_ms)
        return lower_timeframe[start_idx:end_idx]

    while i <= last_entry_index:
        if allowed_entry_days is not None:
            entry_day = datetime.fromtimestamp(entry[i].close_time / 1000, UTC).date().isoformat()
            if entry_day not in allowed_entry_days:
                i += 1
                continue

        close_time_ms = entry[i].close_time

        # Windowed slices: use the same horizon as the live scanner (constant memory/time per step).
        entry_start = max(0, i - max(1, int(entry_window_bars)) + 1)
        entry_slice = entry[entry_start : i + 1]

        trend_end = bisect_right(trend_close_times, close_time_ms) - 1
        if trend_end < 0:
            i += 1
            continue
        trend_start = max(0, trend_end - max(1, int(trend_window_bars)) + 1)
        trend_slice = trend[trend_start : trend_end + 1]
        if len(trend_slice) < 120:
            i += 1
            continue

        market_slice = None
        if market_trend is not None and market_trend_close_times is not None:
            market_end = bisect_right(market_trend_close_times, close_time_ms) - 1
            if market_end >= 0:
                market_start = max(0, market_end - max(1, int(trend_window_bars)) + 1)
                market_slice = market_trend[market_start : market_end + 1]

        market_entry_slice = None
        if market_entry is not None and market_entry_close_times is not None:
            market_entry_end = bisect_right(market_entry_close_times, close_time_ms) - 1
            if market_entry_end >= 0:
                market_entry_start = max(0, market_entry_end - max(1, int(entry_window_bars)) + 1)
                market_entry_slice = market_entry[market_entry_start : market_entry_end + 1]

        candidate = evaluate_observation_candidate(
            symbol=symbol,
            trend_candles=trend_slice,
            entry_candles=entry_slice,
            expected_hold_hours=hold_hours,
            min_hold_hours=min_hold_hours,
            max_hold_hours=max_hold_hours,
            dynamic_hold=dynamic_hold,
            target_rr=target_rr,
            expires_after_minutes=expire_minutes,
            market_candles=market_slice,
            market_entry_candles=market_entry_slice,
            weekly_filter=weekly_filter,
        )
        if allowed_directions is not None and candidate.direction not in allowed_directions:
            i += 1
            continue
        required_score = _threshold_for_candidate(candidate, score_threshold, direction_thresholds)
        if candidate.score < required_score or candidate.direction not in {"做多观察", "做空观察"}:
            i += 1
            continue

        candidate_close_time = close_time_ms
        if _has_conflicting_market_signal(
            candidate=candidate,
            market_trend_candles=market_slice,
            market_entry_candles=market_entry_slice,
            candidate_close_time_ms=candidate_close_time,
            hold_hours=hold_hours,
            target_rr=target_rr,
            expire_minutes=expire_minutes,
            conflict_threshold=conflict_threshold,
            conflict_minutes=conflict_minutes,
        ):
            i += 1
            continue

        pending = _between(candidate_close_time, candidate_close_time + expire_minutes * 60_000)
        confirmed = _find_confirmation_entry(
            candidate=candidate,
            confirmation_candles=pending,
            candidate_close_time_ms=candidate_close_time,
            confirm_minutes=confirm_minutes,
            expire_minutes=expire_minutes,
        )
        if confirmed is None:
            i += max(1, int(round(expire_minutes / 15)))
            continue

        planned_hold_hours = candidate.expected_hold_hours if dynamic_hold else hold_hours
        planned_hold_ms = int(planned_hold_hours * 60 * 60_000)
        future = _between(confirmed.close_time, confirmed.close_time + planned_hold_ms)
        if not future:
            break

        risk = abs(confirmed.close - candidate.stop_loss)
        if candidate.direction == "做多观察":
            take_profit = confirmed.close + target_rr * risk
        else:
            take_profit = confirmed.close - target_rr * risk

        trades.append(
            _simulate_exit(
                symbol=symbol.strip().upper(),
                direction=candidate.direction,
                score=candidate.score,
                entry_candle=confirmed,
                future_candles=future,
                stop_loss=candidate.stop_loss,
                take_profit=take_profit,
                fee_rate=fee_rate,
                funding_rate_8h=funding_rate_8h,
                time_stop_minutes=time_stop_minutes,
                min_progress_r=min_progress_r,
                r_trailing_enabled=r_trailing_enabled,
                trailing_trigger_pct=trailing_trigger_pct,
                trailing_lock_pct=trailing_lock_pct,
                planned_hold_hours=planned_hold_hours,
            )
        )
        i += cooldown if cooldown_bars is not None else max(1, int(round(planned_hold_hours * 60 / 15)))

    return BacktestSummary(
        symbol=symbol.strip().upper(),
        days=days,
        score_threshold=score_threshold,
        hold_hours=hold_hours,
        target_rr=target_rr,
        trades=tuple(trades),
        funding_rate_8h=funding_rate_8h,
    )


async def run_observation_backtest(
    *,
    symbol: str,
    days: int = 30,
    score_threshold: int = 70,
    hold_hours: float = 2.0,
    fee_rate: float = 0.0005,
    target_rr: float = 2.0,
    confirm_minutes: int = 10,
    expire_minutes: int = 20,
    funding_rate_8h: float = 0.0,
    time_stop_minutes: int = 45,
    min_progress_r: float = 0.35,
    r_trailing_enabled: bool = True,
    trailing_trigger_pct: float = 0.03,
    trailing_lock_pct: float = 0.015,
    weekly_filter: bool = True,
    dynamic_hold: bool = True,
    min_hold_hours: float = 1.0,
    max_hold_hours: float = 4.0,
    allowed_entry_days: set[str] | None = None,
    allowed_directions: set[str] | None = None,
    direction_thresholds: dict[tuple[str, str], int] | None = None,
    conflict_threshold: int = 90,
    conflict_minutes: int = 60,
) -> BacktestSummary:
    end = datetime.now(UTC)
    start = end - timedelta(days=days + 8)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    trend_klines = await fetch_usdm_klines_range(
        symbol=symbol,
        interval="1h",
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    )
    entry_klines = await fetch_usdm_klines_range(
        symbol=symbol,
        interval="15m",
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    )
    confirmation_klines = await fetch_usdm_klines_range(
        symbol=symbol,
        interval="5m",
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    )
    market_trend_klines = []
    market_entry_klines = []
    if symbol.strip().upper() != "BTCUSDT":
        market_trend_klines = await fetch_usdm_klines_range(
            symbol="BTCUSDT",
            interval="1h",
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        market_entry_klines = await fetch_usdm_klines_range(
            symbol="BTCUSDT",
            interval="15m",
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
    cutoff_ms = int((end - timedelta(days=days)).timestamp() * 1000)
    trend_candles = [c for c in _to_candles(trend_klines) if c.open_time < end_ms]
    entry_candles = [c for c in _to_candles(entry_klines) if c.open_time >= cutoff_ms and c.open_time < end_ms]
    warmup_entry = [c for c in _to_candles(entry_klines) if c.open_time < cutoff_ms][-140:]
    confirmation_candles = [c for c in _to_candles(confirmation_klines) if c.open_time < end_ms]
    market_trend_candles = [c for c in _to_candles(market_trend_klines) if c.open_time < end_ms] or None
    market_entry_candles = [c for c in _to_candles(market_entry_klines) if c.open_time < end_ms] or None

    return run_observation_backtest_from_candles(
        symbol=symbol,
        trend_candles=trend_candles,
        entry_candles=warmup_entry + entry_candles,
        confirmation_candles=confirmation_candles,
        market_trend_candles=market_trend_candles,
        market_entry_candles=market_entry_candles,
        score_threshold=score_threshold,
        hold_hours=hold_hours,
        fee_rate=fee_rate,
        target_rr=target_rr,
        confirm_minutes=confirm_minutes,
        expire_minutes=expire_minutes,
        funding_rate_8h=funding_rate_8h,
        time_stop_minutes=time_stop_minutes,
        min_progress_r=min_progress_r,
        r_trailing_enabled=r_trailing_enabled,
        trailing_trigger_pct=trailing_trigger_pct,
        trailing_lock_pct=trailing_lock_pct,
        weekly_filter=weekly_filter,
        dynamic_hold=dynamic_hold,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        allowed_entry_days=allowed_entry_days,
        allowed_directions=allowed_directions,
        direction_thresholds=direction_thresholds,
        conflict_threshold=conflict_threshold,
        conflict_minutes=conflict_minutes,
        days=days,
        start_index=len(warmup_entry),
    )


def export_backtest_trades_csv(summaries: Sequence[BacktestSummary], path: str | Path) -> Path:
    """
    Export backtest trades with Excel-friendly headers and readable timestamps.

    Notes:
    - We keep timestamps in both UTC and local time (based on the machine's timezone) to avoid confusion.
    - We intentionally DO NOT include raw millisecond timestamps to prevent Excel from displaying 1.78E+12.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "交易对",
                "方向",
                "评分",
                "入场时间(本地)",
                "出场时间(本地)",
                "入场时间(UTC)",
                "出场时间(UTC)",
                "入场价",
                "出场价",
                "止损价",
                "TP1(1R)",
                "目标价",
                "移动止损触发",
                "移动止损价",
                "计划持仓(小时)",
                "资金费率成本(%)",
                "TP1已触发",
                "出场原因",
                "收益率(%)",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for trade in summary.trades:
                entry_utc = datetime.fromtimestamp(trade.entry_time_ms / 1000, UTC)
                exit_utc = datetime.fromtimestamp(trade.exit_time_ms / 1000, UTC)
                # Avoid depending on tzdata on Windows: use the host machine's local timezone.
                entry_local = entry_utc.astimezone()
                exit_local = exit_utc.astimezone()

                entry_utc_text = entry_utc.isoformat(sep=" ", timespec="seconds")
                exit_utc_text = exit_utc.isoformat(sep=" ", timespec="seconds")
                entry_local_text = entry_local.isoformat(sep=" ", timespec="seconds")
                exit_local_text = exit_local.isoformat(sep=" ", timespec="seconds")

                writer.writerow(
                    {
                        "交易对": trade.symbol,
                        "方向": trade.direction,
                        "评分": trade.score,
                        "入场时间(本地)": entry_local_text,
                        "出场时间(本地)": exit_local_text,
                        "入场时间(UTC)": entry_utc_text,
                        "出场时间(UTC)": exit_utc_text,
                        "入场价": trade.entry_price,
                        "出场价": trade.exit_price,
                        "止损价": trade.stop_loss,
                        "TP1(1R)": trade.tp1_price,
                        "目标价": trade.final_target_price,
                        "移动止损触发": trade.trailing_stop_activated,
                        "移动止损价": trade.trailing_stop_price,
                        "计划持仓(小时)": trade.planned_hold_hours,
                        "资金费率成本(%)": round(trade.funding_cost_pct * 100, 6),
                        "TP1已触发": trade.tp1_hit,
                        "出场原因": trade.outcome,
                        "收益率(%)": round(trade.pnl_pct * 100, 6),
                    }
                )
    return out
