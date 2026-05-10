from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_signal_bot.backtest import _funding_cost_pct, _partial_pnl_pct, _pnl_pct, _r_trailing_stop_price
from crypto_signal_bot.market import fetch_usdm_klines_range
from crypto_signal_bot.models import Candle, Kline
from crypto_signal_bot.strategies import ObservationCandidate

PENDING = "pending"
BREAKOUT_CONFIRMED = "breakout_confirmed"
WAITING_HOLD_CONFIRMATION = "waiting_hold_confirmation"
# NOTE: v1 used "wait_pullback". We keep backward compatibility on load, but new ledgers use "waiting_pullback".
WAITING_PULLBACK = "waiting_pullback"
LEGACY_WAIT_PULLBACK = "wait_pullback"
# Backward-compat import name (older code/tests/CLI expect WAIT_PULLBACK).
WAIT_PULLBACK = WAITING_PULLBACK
TRIGGERED = "triggered"
OPEN = "open"
CLOSED = "closed"
EXPIRED = "expired"
CANCELLED = "cancelled"
SKIPPED = "skipped"
ACTIVE_STATUSES = {
    PENDING,
    BREAKOUT_CONFIRMED,
    WAITING_HOLD_CONFIRMATION,
    WAITING_PULLBACK,
    TRIGGERED,
    OPEN,
}

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS

# 5m confirmation candle quality filter: avoid tiny wick pokes through trigger.
MIN_CONFIRM_BODY_RATIO = 0.45
MIN_CONFIRM_CLOSE_POS = 0.60


@dataclass(frozen=True)
class PaperTrade:
    paper_id: str
    symbol: str
    direction: str
    score: int
    level: str
    status: str
    signal_time_ms: int
    signal_price: float
    confirm_until_ms: int
    expires_at_ms: int
    trigger_price: float
    stop_loss: float
    signal_take_profit_1: float
    signal_take_profit_2: float
    target_rr: float
    planned_hold_hours: float
    reasons: tuple[str, ...] = ()
    trigger_time_ms: int | None = None
    trigger_fill_price: float = 0.0
    trigger_confirmed: bool = False
    entry_time_ms: int | None = None
    entry_price: float = 0.0
    exit_time_ms: int | None = None
    exit_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    final_target_price: float = 0.0
    active_stop_price: float = 0.0
    tp1_hit: bool = False
    tp1_time_ms: int | None = None
    tp1_hit_price: float = 0.0
    tp2_hit: bool = False
    tp2_time_ms: int | None = None
    tp2_hit_price: float = 0.0
    moved_stop_to_breakeven: bool = False
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    final_tp_r: float = 3.0
    initial_risk_pct: float = 0.0
    realized_r_multiple: float = 0.0
    partial_close_pct_at_tp1: float = 0.5
    trailing_stop_price: float = 0.0
    trailing_stop_activated: bool = False
    best_progress_r: float = 0.0
    outcome: str = ""
    pnl_pct: float = 0.0
    funding_cost_pct: float = 0.0
    last_checked_ms: int = 0
    last_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    # MFE/MAE tracking (since entry)
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    max_favorable_pnl_pct: float = 0.0
    max_adverse_pnl_pct: float = 0.0
    max_favorable_r: float = 0.0
    max_adverse_r: float = 0.0
    time_to_mfe_minutes: float | None = None
    time_to_mae_minutes: float | None = None
    # Scoring / hard-filter audit (for analyzing score inflation and filter effectiveness)
    legacy_score: int = 0
    hard_filter_passed: bool = False
    failed_hard_filters: tuple[str, ...] = ()
    raw_score: float = 0.0
    final_score: int = 0
    score_components_json: str = ""
    market_regime: str = "unknown"
    market_regime_confidence: int = 0
    no_trade_zone: bool = False
    market_regime_reasons: tuple[str, ...] = ()
    # No-chase / anti-FOMO entry guard (evaluated at trigger time; may switch to WAIT_PULLBACK or SKIPPED).
    no_chase_passed: bool | None = None
    entry_slippage_r: float | None = None
    trigger_candle_atr_multiple: float | None = None
    wait_pullback: bool = False
    skip_reason: str = ""
    # Breakout confirmation config/state (paper/backtest entry only; does not affect Telegram "观察信号" push).
    breakout_confirmation_mode: str = "close_and_hold"  # close_only / close_and_hold / close_and_pullback / legacy
    confirmation_candle: str = "5m"
    hold_confirmation_candles_required: int = 1
    hold_confirmation_candles_seen: int = 0
    pullback_tolerance_r: float = 0.25
    pullback_expire_minutes: int = 30
    pullback_expires_at_ms: int | None = None
    # Status transition audit: append an event whenever status changes.
    status_events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PaperSyncResult:
    trades: tuple[PaperTrade, ...]
    changed_count: int
    errors: tuple[str, ...] = ()


KlineRangeFetcher = Callable[..., Awaitable[Sequence[Kline]]]


def now_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def load_paper_trades(path: str | Path) -> list[PaperTrade]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    items = raw.get("trades", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"虚拟盘账本格式错误：{ledger_path}")
    return [_paper_trade_from_dict(item) for item in items]


def save_paper_trades(path: str | Path, trades: Sequence[PaperTrade]) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "trades": [asdict(trade) for trade in trades],
    }
    ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger_path


def record_signal_candidates(
    candidates: Sequence[ObservationCandidate],
    path: str | Path,
    *,
    signal_time_ms: int | None = None,
    confirm_minutes: int = 10,
    entry_mode: str = "confirm",
    cooldown_minutes: int = 60,
    dedupe_minutes: int = 60,
    breakout_confirmation_mode: str = "close_and_hold",
    confirmation_candle: str = "5m",
    hold_confirmation_candles: int = 1,
    pullback_tolerance_r: float = 0.25,
    pullback_expire_minutes: int = 30,
    tp1_r: float = 1.0,
    tp2_r: float = 2.0,
    final_tp_r: float = 3.0,
    partial_close_pct_at_tp1: float = 0.5,
) -> list[PaperTrade]:
    signal_at = signal_time_ms if signal_time_ms is not None else now_ms()
    trades = load_paper_trades(path)
    recorded: list[PaperTrade] = []
    for candidate in candidates:
        if candidate.direction not in {"做多观察", "做空观察"}:
            continue
        if _is_in_cooldown(trades, candidate, signal_at, cooldown_minutes):
            continue
        if _has_duplicate_active_trade(trades, candidate, signal_at, dedupe_minutes):
            continue
        trade = _paper_trade_from_candidate(
            candidate,
            signal_time_ms=signal_at,
            confirm_minutes=confirm_minutes,
            entry_mode=entry_mode,
            breakout_confirmation_mode=breakout_confirmation_mode,
            confirmation_candle=confirmation_candle,
            hold_confirmation_candles=hold_confirmation_candles,
            pullback_tolerance_r=pullback_tolerance_r,
            pullback_expire_minutes=pullback_expire_minutes,
            tp1_r=tp1_r,
            tp2_r=tp2_r,
            final_tp_r=final_tp_r,
            partial_close_pct_at_tp1=partial_close_pct_at_tp1,
        )
        trades.append(trade)
        recorded.append(trade)
    if recorded:
        save_paper_trades(path, trades)
    return recorded


def record_signal_candidate(
    candidate: ObservationCandidate,
    path: str | Path,
    *,
    signal_time_ms: int | None = None,
    confirm_minutes: int = 10,
    entry_mode: str = "confirm",
    cooldown_minutes: int = 60,
    dedupe_minutes: int = 60,
    breakout_confirmation_mode: str = "close_and_hold",
    confirmation_candle: str = "5m",
    hold_confirmation_candles: int = 1,
    pullback_tolerance_r: float = 0.25,
    pullback_expire_minutes: int = 30,
    tp1_r: float = 1.0,
    tp2_r: float = 2.0,
    final_tp_r: float = 3.0,
    partial_close_pct_at_tp1: float = 0.5,
) -> PaperTrade | None:
    recorded = record_signal_candidates(
        [candidate],
        path,
        signal_time_ms=signal_time_ms,
        confirm_minutes=confirm_minutes,
        entry_mode=entry_mode,
        cooldown_minutes=cooldown_minutes,
        dedupe_minutes=dedupe_minutes,
        breakout_confirmation_mode=breakout_confirmation_mode,
        confirmation_candle=confirmation_candle,
        hold_confirmation_candles=hold_confirmation_candles,
        pullback_tolerance_r=pullback_tolerance_r,
        pullback_expire_minutes=pullback_expire_minutes,
        tp1_r=tp1_r,
        tp2_r=tp2_r,
        final_tp_r=final_tp_r,
        partial_close_pct_at_tp1=partial_close_pct_at_tp1,
    )
    return recorded[0] if recorded else None


def update_paper_trade_with_candles(
    trade: PaperTrade,
    candles: Sequence[Candle],
    *,
    now_time_ms: int,
    fee_rate: float = 0.0005,
    funding_rate_8h: float = 0.0001,
    time_stop_minutes: int = 45,
    min_progress_r: float = 0.35,
    r_trailing_enabled: bool = True,
    trailing_trigger_pct: float = 0.03,
    trailing_lock_pct: float = 0.015,
    cancel_if_stop_before_trigger: bool = True,
    move_stop_to_breakeven_after_tp1: bool = True,
    max_entry_slippage_r: float = 0.5,
    max_trigger_candle_atr_multiple: float = 1.2,
    wait_pullback_if_chased: bool = True,
    breakout_confirmation_mode: str = "close_and_hold",
    hold_confirmation_candles: int = 1,
    pullback_tolerance_r: float = 0.25,
    pullback_expire_minutes: int = 30,
) -> PaperTrade:
    if trade.status not in ACTIVE_STATUSES:
        return trade

    ordered = sorted((c for c in candles if c.is_closed), key=lambda c: c.close_time)
    # Keep active trades aligned with the current runtime config (so CLI flags apply to existing pending signals).
    configured = _apply_breakout_confirmation_config(
        trade,
        breakout_confirmation_mode=breakout_confirmation_mode,
        hold_confirmation_candles=hold_confirmation_candles,
        pullback_tolerance_r=pullback_tolerance_r,
        pullback_expire_minutes=pullback_expire_minutes,
    )
    if configured.status in {PENDING, BREAKOUT_CONFIRMED}:
        return _update_pending_trade(
            configured,
            ordered,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
            wait_pullback_if_chased=wait_pullback_if_chased,
        )
    if configured.status == WAITING_HOLD_CONFIRMATION:
        return _update_hold_confirmation_trade(
            configured,
            ordered,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
            wait_pullback_if_chased=wait_pullback_if_chased,
        )
    if configured.status == WAITING_PULLBACK:
        return _update_wait_pullback_trade(
            configured,
            ordered,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            pullback_tolerance_r=pullback_tolerance_r,
        )
    # TRIGGERED is a transient state for "entry just confirmed". Next sync step materializes it as OPEN.
    if configured.status == TRIGGERED:
        open_time = int(configured.entry_time_ms or now_time_ms)
        materialized = _enter_status(configured, OPEN, time_ms=open_time, reason="materialize_open")
    else:
        materialized = configured
    updated = _advance_open_trade(
        materialized,
        ordered,
        now_time_ms=now_time_ms,
        fee_rate=fee_rate,
        funding_rate_8h=funding_rate_8h,
        time_stop_minutes=time_stop_minutes,
        min_progress_r=min_progress_r,
        r_trailing_enabled=r_trailing_enabled,
        trailing_trigger_pct=trailing_trigger_pct,
        trailing_lock_pct=trailing_lock_pct,
        move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
    )
    return _mark_to_market(updated, ordered)


async def sync_paper_trades(
    path: str | Path,
    *,
    fee_rate: float = 0.0005,
    funding_rate_8h: float = 0.0001,
    time_stop_minutes: int = 45,
    min_progress_r: float = 0.35,
    r_trailing_enabled: bool = True,
    trailing_trigger_pct: float = 0.03,
    trailing_lock_pct: float = 0.015,
    cancel_if_stop_before_trigger: bool = True,
    move_stop_to_breakeven_after_tp1: bool = True,
    max_entry_slippage_r: float = 0.5,
    max_trigger_candle_atr_multiple: float = 1.2,
    wait_pullback_if_chased: bool = True,
    breakout_confirmation_mode: str = "close_and_hold",
    hold_confirmation_candles: int = 1,
    pullback_tolerance_r: float = 0.25,
    pullback_expire_minutes: int = 30,
    interval: str = "5m",
    fetcher: KlineRangeFetcher = fetch_usdm_klines_range,
    now_time_ms: int | None = None,
) -> PaperSyncResult:
    trades = load_paper_trades(path)
    if not trades:
        return PaperSyncResult((), 0)

    current_ms = now_time_ms if now_time_ms is not None else now_ms()
    active_by_symbol: dict[str, list[PaperTrade]] = defaultdict(list)
    for trade in trades:
        if trade.status in ACTIVE_STATUSES:
            active_by_symbol[trade.symbol].append(trade)

    candles_by_symbol: dict[str, list[Candle]] = {}
    errors: list[str] = []
    for symbol, active_trades in active_by_symbol.items():
        start_ms = min(_fetch_start_ms(trade) for trade in active_trades)
        try:
            klines = await fetcher(
                symbol=symbol,
                interval=interval,
                start_time_ms=start_ms,
                end_time_ms=current_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime logging path
            errors.append(f"{symbol} 行情同步失败：{exc}")
            continue
        candles_by_symbol[symbol] = [Candle.from_kline(kline) for kline in klines]

    changed = 0
    updated: list[PaperTrade] = []
    for trade in trades:
        candles = candles_by_symbol.get(trade.symbol, [])
        new_trade = update_paper_trade_with_candles(
            trade,
            candles,
            now_time_ms=current_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
            wait_pullback_if_chased=wait_pullback_if_chased,
            breakout_confirmation_mode=breakout_confirmation_mode,
            hold_confirmation_candles=hold_confirmation_candles,
            pullback_tolerance_r=pullback_tolerance_r,
            pullback_expire_minutes=pullback_expire_minutes,
        )
        if new_trade != trade:
            changed += 1
        updated.append(new_trade)

    if changed:
        save_paper_trades(path, updated)
    return PaperSyncResult(tuple(updated), changed, tuple(errors))


def paper_summary_zh(trades: Sequence[PaperTrade]) -> str:
    total = len(trades)
    pending = sum(1 for trade in trades if trade.status == PENDING)
    wait_pullback = sum(1 for trade in trades if trade.status == WAIT_PULLBACK)
    triggered = sum(1 for trade in trades if trade.status == TRIGGERED)
    open_count = sum(1 for trade in trades if trade.status == OPEN)
    expired = sum(1 for trade in trades if trade.status == EXPIRED)
    cancelled = sum(1 for trade in trades if trade.status == CANCELLED)
    skipped = sum(1 for trade in trades if trade.status == SKIPPED)
    closed = [trade for trade in trades if trade.status == CLOSED]
    wins = sum(1 for trade in closed if trade.pnl_pct > 0)
    losses = sum(1 for trade in closed if trade.pnl_pct <= 0)
    total_return = sum(trade.pnl_pct for trade in closed) * 100
    win_rate = wins / len(closed) * 100 if closed else 0.0
    funding_cost = sum(trade.funding_cost_pct for trade in closed) * 100

    lines = [
        "虚拟盘记录摘要",
        f"总记录：{total} 笔",
        (
            f"待触发：{pending} 笔，等待回踩：{wait_pullback} 笔，已触发：{triggered} 笔，持仓中：{open_count} 笔，"
            f"已平仓：{len(closed)} 笔，未触发过期：{expired} 笔，已取消：{cancelled} 笔，已跳过：{skipped} 笔"
        ),
        f"已平仓胜率：{win_rate:.2f}%（盈利 {wins} / 亏损 {losses}）",
        f"已平仓累计收益率：{total_return:.2f}%",
        f"资金费率成本估算：{funding_cost:.3f}%",
    ]
    active = [trade for trade in trades if trade.status in ACTIVE_STATUSES]
    if active:
        lines.append("")
        lines.append("当前未结束记录：")
        for trade in sorted(active, key=lambda item: item.signal_time_ms)[-8:]:
            lines.append(_paper_trade_line_zh(trade))
    latest_closed = sorted(closed, key=lambda item: item.exit_time_ms or 0)[-5:]
    if latest_closed:
        lines.append("")
        lines.append("最近平仓：")
        for trade in latest_closed:
            lines.append(_paper_trade_line_zh(trade))
    return "\n".join(lines)


def export_paper_trades(path: str | Path, trades: Sequence[PaperTrade]) -> Path:
    out = Path(path)
    if out.suffix.lower() == ".xlsx":
        try:
            return _export_paper_trades_xlsx(out, trades)
        except ModuleNotFoundError:
            csv_path = out.with_suffix(".csv")
            return _export_paper_trades_csv(csv_path, trades)
    return _export_paper_trades_csv(out, trades)


def _export_paper_trades_csv(path: Path, trades: Sequence[PaperTrade]) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "paper_id",
        "symbol",
        "direction",
        "status",
        "score",
        "legacy_score",
        "hard_filter_passed",
        "failed_hard_filters",
        "raw_score",
        "final_score",
        "score_components_json",
        "market_regime",
        "market_regime_confidence",
        "no_trade_zone",
        "market_regime_reasons",
        "level",
        "signal_time",
        "signal_price",
        "signal_tp1_price",
        "signal_tp2_price",
        "trigger_time",
        "trigger_fill_price",
        "trigger_confirmed",
        "no_chase_passed",
        "entry_slippage_r",
        "trigger_candle_atr_multiple",
        "wait_pullback",
        "skip_reason",
        "entry_time",
        "entry_price",
        "last_price",
        "unrealized_pnl_pct",
        "stop_loss",
        "tp1_price",
        "tp2_price",
        "final_target_price",
        "active_stop_price",
        "tp1_hit",
        "tp1_hit_time",
        "tp1_hit_price",
        "tp2_hit",
        "tp2_hit_time",
        "tp2_hit_price",
        "moved_stop_to_breakeven",
        "tp1_r",
        "tp2_r",
        "final_tp_r",
        "partial_close_pct_at_tp1",
        "initial_risk_pct",
        "realized_r_multiple",
        "trailing_stop_activated",
        "trailing_stop_price",
        "outcome",
        "exit_time",
        "exit_price",
        "pnl_pct",
        "funding_cost_pct",
        "max_favorable_price",
        "max_adverse_price",
        "max_favorable_pnl_pct",
        "max_adverse_pnl_pct",
        "max_favorable_r",
        "max_adverse_r",
        "time_to_mfe_minutes",
        "time_to_mae_minutes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in trades:
            w.writerow(
                {
                    "paper_id": t.paper_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "status": t.status,
                    "score": t.score,
                    "legacy_score": t.legacy_score,
                    "hard_filter_passed": t.hard_filter_passed,
                    "failed_hard_filters": ",".join(t.failed_hard_filters) if t.failed_hard_filters else "",
                    "raw_score": round(float(t.raw_score), 6),
                    "final_score": t.final_score or t.score,
                    "score_components_json": t.score_components_json,
                    "market_regime": t.market_regime,
                    "market_regime_confidence": int(t.market_regime_confidence),
                    "no_trade_zone": t.no_trade_zone,
                    "market_regime_reasons": " | ".join(t.market_regime_reasons) if t.market_regime_reasons else "",
                    "level": t.level,
                    "signal_time": _format_ms(t.signal_time_ms),
                    "signal_price": t.signal_price,
                    "signal_tp1_price": t.signal_take_profit_1,
                    "signal_tp2_price": t.signal_take_profit_2,
                    "trigger_time": _format_ms(t.trigger_time_ms),
                    "trigger_fill_price": t.trigger_fill_price,
                    "trigger_confirmed": t.trigger_confirmed,
                    "no_chase_passed": "" if t.no_chase_passed is None else bool(t.no_chase_passed),
                    "entry_slippage_r": "" if t.entry_slippage_r is None else round(float(t.entry_slippage_r), 6),
                    "trigger_candle_atr_multiple": ""
                    if t.trigger_candle_atr_multiple is None
                    else round(float(t.trigger_candle_atr_multiple), 6),
                    "wait_pullback": bool(t.wait_pullback),
                    "skip_reason": t.skip_reason,
                    "entry_time": _format_ms(t.entry_time_ms),
                    "entry_price": t.entry_price,
                    "last_price": t.last_price,
                    "unrealized_pnl_pct": round(t.unrealized_pnl_pct * 100, 6),
                    "stop_loss": t.stop_loss,
                    "tp1_price": t.tp1_price,
                    "tp2_price": t.tp2_price,
                    "final_target_price": t.final_target_price,
                    "active_stop_price": t.active_stop_price,
                    "tp1_hit": t.tp1_hit,
                    "tp1_hit_time": _format_ms(t.tp1_time_ms),
                    "tp1_hit_price": t.tp1_hit_price,
                    "tp2_hit": t.tp2_hit,
                    "tp2_hit_time": _format_ms(t.tp2_time_ms),
                    "tp2_hit_price": t.tp2_hit_price,
                    "moved_stop_to_breakeven": t.moved_stop_to_breakeven,
                    "tp1_r": round(float(t.tp1_r), 6),
                    "tp2_r": round(float(t.tp2_r), 6),
                    "final_tp_r": round(float(t.final_tp_r), 6),
                    "partial_close_pct_at_tp1": round(float(t.partial_close_pct_at_tp1), 6),
                    "initial_risk_pct": round(t.initial_risk_pct * 100, 6),
                    "realized_r_multiple": round(t.realized_r_multiple, 6),
                    "trailing_stop_activated": t.trailing_stop_activated,
                    "trailing_stop_price": t.trailing_stop_price,
                    "outcome": t.outcome,
                    "exit_time": _format_ms(t.exit_time_ms),
                    "exit_price": t.exit_price,
                    "pnl_pct": round(t.pnl_pct * 100, 6),
                    "funding_cost_pct": round(t.funding_cost_pct * 100, 6),
                    "max_favorable_price": t.max_favorable_price,
                    "max_adverse_price": t.max_adverse_price,
                    "max_favorable_pnl_pct": round(t.max_favorable_pnl_pct * 100, 6),
                    "max_adverse_pnl_pct": round(t.max_adverse_pnl_pct * 100, 6),
                    "max_favorable_r": round(t.max_favorable_r, 6),
                    "max_adverse_r": round(t.max_adverse_r, 6),
                    "time_to_mfe_minutes": round(float(t.time_to_mfe_minutes), 6) if t.time_to_mfe_minutes is not None else "",
                    "time_to_mae_minutes": round(float(t.time_to_mae_minutes), 6) if t.time_to_mae_minutes is not None else "",
                }
            )
    return path


def _export_paper_trades_xlsx(path: Path, trades: Sequence[PaperTrade]) -> Path:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "paper_trades"

    headers = [
        "paper_id",
        "symbol",
        "direction",
        "status",
        "score",
        "legacy_score",
        "hard_filter_passed",
        "failed_hard_filters",
        "raw_score",
        "final_score",
        "score_components_json",
        "market_regime",
        "market_regime_confidence",
        "no_trade_zone",
        "market_regime_reasons",
        "level",
        "signal_time",
        "signal_price",
        "signal_tp1_price",
        "signal_tp2_price",
        "trigger_time",
        "trigger_fill_price",
        "trigger_confirmed",
        "no_chase_passed",
        "entry_slippage_r",
        "trigger_candle_atr_multiple",
        "wait_pullback",
        "skip_reason",
        "entry_time",
        "entry_price",
        "last_price",
        "unrealized_pnl_pct(%)",
        "stop_loss",
        "tp1_price",
        "tp2_price",
        "final_target_price",
        "active_stop_price",
        "tp1_hit",
        "tp1_hit_time",
        "tp1_hit_price",
        "tp2_hit",
        "tp2_hit_time",
        "tp2_hit_price",
        "moved_stop_to_breakeven",
        "tp1_r",
        "tp2_r",
        "final_tp_r",
        "partial_close_pct_at_tp1",
        "initial_risk_pct(%)",
        "realized_r_multiple",
        "trailing_stop_activated",
        "trailing_stop_price",
        "outcome",
        "exit_time",
        "exit_price",
        "pnl_pct(%)",
        "funding_cost_pct(%)",
        "max_favorable_price",
        "max_adverse_price",
        "max_favorable_pnl_pct(%)",
        "max_adverse_pnl_pct(%)",
        "max_favorable_r",
        "max_adverse_r",
        "time_to_mfe_minutes",
        "time_to_mae_minutes",
    ]
    ws.append(headers)
    for t in trades:
        ws.append(
            [
                t.paper_id,
                t.symbol,
                t.direction,
                t.status,
                t.score,
                t.legacy_score,
                int(t.hard_filter_passed),
                ",".join(t.failed_hard_filters) if t.failed_hard_filters else "",
                float(t.raw_score),
                t.final_score or t.score,
                t.score_components_json,
                t.market_regime,
                int(t.market_regime_confidence),
                int(t.no_trade_zone),
                " | ".join(t.market_regime_reasons) if t.market_regime_reasons else "",
                t.level,
                _format_ms(t.signal_time_ms),
                t.signal_price,
                t.signal_take_profit_1,
                t.signal_take_profit_2,
                _format_ms(t.trigger_time_ms),
                t.trigger_fill_price,
                int(t.trigger_confirmed),
                "" if t.no_chase_passed is None else int(bool(t.no_chase_passed)),
                float(t.entry_slippage_r) if t.entry_slippage_r is not None else "",
                float(t.trigger_candle_atr_multiple) if t.trigger_candle_atr_multiple is not None else "",
                int(bool(t.wait_pullback)),
                t.skip_reason,
                _format_ms(t.entry_time_ms),
                t.entry_price,
                t.last_price,
                round(t.unrealized_pnl_pct * 100, 6),
                t.stop_loss,
                t.tp1_price,
                t.tp2_price,
                t.final_target_price,
                t.active_stop_price,
                int(t.tp1_hit),
                _format_ms(t.tp1_time_ms),
                t.tp1_hit_price,
                int(t.tp2_hit),
                _format_ms(t.tp2_time_ms),
                t.tp2_hit_price,
                int(t.moved_stop_to_breakeven),
                round(float(t.tp1_r), 6),
                round(float(t.tp2_r), 6),
                round(float(t.final_tp_r), 6),
                round(float(t.partial_close_pct_at_tp1), 6),
                round(t.initial_risk_pct * 100, 6),
                round(t.realized_r_multiple, 6),
                int(t.trailing_stop_activated),
                t.trailing_stop_price,
                t.outcome,
                _format_ms(t.exit_time_ms),
                t.exit_price,
                round(t.pnl_pct * 100, 6),
                round(t.funding_cost_pct * 100, 6),
                t.max_favorable_price,
                t.max_adverse_price,
                round(t.max_favorable_pnl_pct * 100, 6),
                round(t.max_adverse_pnl_pct * 100, 6),
                round(t.max_favorable_r, 6),
                round(t.max_adverse_r, 6),
                round(float(t.time_to_mfe_minutes), 6) if t.time_to_mfe_minutes is not None else "",
                round(float(t.time_to_mae_minutes), 6) if t.time_to_mae_minutes is not None else "",
            ]
        )
    wb.save(path)
    return path


def _paper_trade_from_dict(item: Any) -> PaperTrade:
    if not isinstance(item, dict):
        raise ValueError("虚拟盘账本存在非法记录")
    names = {field.name for field in fields(PaperTrade)}
    values = {name: item[name] for name in names if name in item}
    # Backward compatibility: older ledgers used "wait_pullback" for the pullback-wait state.
    if values.get("status") == LEGACY_WAIT_PULLBACK:
        values["status"] = WAITING_PULLBACK
    if "reasons" in values:
        values["reasons"] = tuple(values["reasons"])
    if "failed_hard_filters" in values:
        values["failed_hard_filters"] = tuple(values["failed_hard_filters"])
    if "market_regime_reasons" in values:
        values["market_regime_reasons"] = tuple(values["market_regime_reasons"])
    if "status_events" in values:
        values["status_events"] = tuple(values["status_events"])
    return PaperTrade(**values)


def _paper_trade_from_candidate(
    candidate: ObservationCandidate,
    *,
    signal_time_ms: int,
    confirm_minutes: int,
    entry_mode: str,
    breakout_confirmation_mode: str,
    confirmation_candle: str,
    hold_confirmation_candles: int,
    pullback_tolerance_r: float,
    pullback_expire_minutes: int,
    tp1_r: float,
    tp2_r: float,
    final_tp_r: float,
    partial_close_pct_at_tp1: float,
) -> PaperTrade:
    confirm_until_ms = signal_time_ms + max(1, confirm_minutes) * MINUTE_MS
    expires_at_ms = signal_time_ms + max(1, candidate.expires_after_minutes) * MINUTE_MS
    base = PaperTrade(
        paper_id=_candidate_id(candidate, signal_time_ms),
        symbol=candidate.symbol.strip().upper(),
        direction=candidate.direction,
        score=candidate.score,
        legacy_score=int(getattr(candidate, "legacy_score", 0) or 0),
        hard_filter_passed=bool(getattr(candidate, "hard_filter_passed", False)),
        failed_hard_filters=tuple(getattr(candidate, "failed_hard_filters", ()) or ()),
        raw_score=float(getattr(candidate, "raw_score", 0.0) or 0.0),
        final_score=int(getattr(candidate, "final_score", candidate.score) or candidate.score),
        score_components_json=str(getattr(candidate, "score_components_json", "") or ""),
        market_regime=str(getattr(candidate, "market_regime", "unknown") or "unknown"),
        market_regime_confidence=int(getattr(candidate, "market_regime_confidence", 0) or 0),
        no_trade_zone=bool(getattr(candidate, "no_trade_zone", False)),
        market_regime_reasons=tuple(getattr(candidate, "market_regime_reasons", ()) or ()),
        level=candidate.level,
        status=PENDING,
        signal_time_ms=signal_time_ms,
        signal_price=candidate.current_price,
        confirm_until_ms=confirm_until_ms,
        expires_at_ms=expires_at_ms,
        trigger_price=candidate.trigger_price,
        stop_loss=candidate.stop_loss,
        signal_take_profit_1=candidate.take_profit_1,
        signal_take_profit_2=candidate.take_profit_2,
        target_rr=candidate.target_rr,
        planned_hold_hours=candidate.expected_hold_hours,
        reasons=candidate.reasons,
        active_stop_price=candidate.stop_loss,
        last_checked_ms=signal_time_ms,
        tp1_r=float(tp1_r),
        tp2_r=float(tp2_r),
        final_tp_r=float(final_tp_r),
        partial_close_pct_at_tp1=float(partial_close_pct_at_tp1),
        breakout_confirmation_mode=str(breakout_confirmation_mode or "close_and_hold"),
        confirmation_candle=str(confirmation_candle or "5m"),
        hold_confirmation_candles_required=max(1, int(hold_confirmation_candles)),
        hold_confirmation_candles_seen=0,
        pullback_tolerance_r=float(pullback_tolerance_r),
        pullback_expire_minutes=max(1, int(pullback_expire_minutes)),
        pullback_expires_at_ms=None,
        status_events=(
            {
                "status": PENDING,
                "time_ms": int(signal_time_ms),
                "reason": "signal_created",
            },
        ),
    )
    mode = (entry_mode or "confirm").strip().lower()
    if mode in {"immediate", "now", "market"}:
        # Legacy behavior for A/B comparisons: enter immediately at signal_time using signal_price.
        entry = candidate.current_price
        is_long = candidate.direction == "做多观察"
        risk = abs(entry - candidate.stop_loss)
        risk = max(risk, 1e-12)
        tp1 = entry + risk * float(tp1_r) if is_long else entry - risk * float(tp1_r)
        tp2 = entry + risk * float(tp2_r) if is_long else entry - risk * float(tp2_r)
        final_tp = entry + risk * float(final_tp_r) if is_long else entry - risk * float(final_tp_r)
        risk_pct = (risk / entry) if entry > 0 else 0.0
        return replace(
            base,
            status=OPEN,
            entry_time_ms=signal_time_ms,
            entry_price=entry,
            tp1_price=tp1,
            tp2_price=tp2,
            final_target_price=final_tp,
            active_stop_price=candidate.stop_loss,
            last_price=entry,
            unrealized_pnl_pct=0.0,
            initial_risk_pct=risk_pct,
            max_favorable_price=entry,
            max_adverse_price=entry,
            max_favorable_pnl_pct=0.0,
            max_adverse_pnl_pct=0.0,
            max_favorable_r=0.0,
            max_adverse_r=0.0,
            time_to_mfe_minutes=0.0,
            time_to_mae_minutes=0.0,
        )
    # Default: record a pending signal first; wait for trigger confirmation to enter.
    return base


def _candidate_id(candidate: ObservationCandidate, signal_time_ms: int) -> str:
    raw = "|".join(
        (
            candidate.symbol.strip().upper(),
            candidate.direction,
            str(signal_time_ms // MINUTE_MS),
            f"{candidate.trigger_price:.8f}",
            f"{candidate.stop_loss:.8f}",
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14].upper()


def _has_duplicate_active_trade(
    trades: Sequence[PaperTrade],
    candidate: ObservationCandidate,
    signal_time_ms: int,
    dedupe_minutes: int,
) -> bool:
    max_gap_ms = max(1, dedupe_minutes) * MINUTE_MS
    for trade in trades:
        if trade.status not in ACTIVE_STATUSES:
            continue
        if trade.symbol != candidate.symbol.strip().upper() or trade.direction != candidate.direction:
            continue
        trigger_gap = abs(trade.trigger_price - candidate.trigger_price) / max(candidate.trigger_price, 1e-9)
        stop_gap = abs(trade.stop_loss - candidate.stop_loss) / max(candidate.stop_loss, 1e-9)
        if trigger_gap <= 0.001 and stop_gap <= 0.001 and abs(signal_time_ms - trade.signal_time_ms) <= max_gap_ms:
            return True
    return False


def _is_in_cooldown(
    trades: Sequence[PaperTrade],
    candidate: ObservationCandidate,
    signal_time_ms: int,
    cooldown_minutes: int,
) -> bool:
    if cooldown_minutes <= 0:
        return False
    cooldown_ms = cooldown_minutes * MINUTE_MS
    symbol = candidate.symbol.strip().upper()
    direction = candidate.direction
    last_event = -1
    for trade in trades:
        if trade.symbol != symbol or trade.direction != direction:
            continue
        event_time = trade.exit_time_ms or trade.entry_time_ms or trade.signal_time_ms
        if event_time is not None:
            last_event = max(last_event, int(event_time))
    return last_event >= 0 and (signal_time_ms - last_event) < cooldown_ms


def _fetch_start_ms(trade: PaperTrade) -> int:
    anchor = trade.last_checked_ms or trade.signal_time_ms
    return max(0, anchor - 2 * MINUTE_MS)


def _apply_breakout_confirmation_config(
    trade: PaperTrade,
    *,
    breakout_confirmation_mode: str,
    hold_confirmation_candles: int,
    pullback_tolerance_r: float,
    pullback_expire_minutes: int,
) -> PaperTrade:
    mode = (breakout_confirmation_mode or trade.breakout_confirmation_mode or "close_and_hold").strip().lower()
    if mode not in {"legacy", "close_only", "close_and_hold", "close_and_pullback"}:
        mode = "close_and_hold"
    required = max(1, int(hold_confirmation_candles))
    tolerance = float(pullback_tolerance_r)
    expire_mins = max(1, int(pullback_expire_minutes))
    # We keep pullback_expires_at_ms as runtime state, so don't overwrite it here.
    updated = trade
    if updated.breakout_confirmation_mode != mode:
        updated = replace(updated, breakout_confirmation_mode=mode)
    if updated.hold_confirmation_candles_required != required:
        updated = replace(updated, hold_confirmation_candles_required=required)
    if abs(updated.pullback_tolerance_r - tolerance) > 1e-12:
        updated = replace(updated, pullback_tolerance_r=tolerance)
    if updated.pullback_expire_minutes != expire_mins:
        updated = replace(updated, pullback_expire_minutes=expire_mins)
    return updated


def _enter_status(trade: PaperTrade, new_status: str, *, time_ms: int, reason: str = "") -> PaperTrade:
    if trade.status == new_status:
        return trade
    events = trade.status_events or ()
    return replace(
        trade,
        status=new_status,
        status_events=events
        + (
            {
                "status": new_status,
                "time_ms": int(time_ms),
                "reason": reason,
            },
        ),
    )


def _update_pending_trade(
    trade: PaperTrade,
    candles: Sequence[Candle],
    *,
    now_time_ms: int,
    fee_rate: float,
    funding_rate_8h: float,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    cancel_if_stop_before_trigger: bool,
    move_stop_to_breakeven_after_tp1: bool,
    max_entry_slippage_r: float,
    max_trigger_candle_atr_multiple: float,
    wait_pullback_if_chased: bool,
) -> PaperTrade:
    last_checked = trade.last_checked_ms
    mode = (trade.breakout_confirmation_mode or "close_and_hold").strip().lower()

    working = trade
    # BREAKOUT_CONFIRMED is a transitional state: once we have the breakout close/touch,
    # we advance into the next confirmation stage based on breakout_confirmation_mode.
    if working.status == BREAKOUT_CONFIRMED:
        anchor_ms = int(working.trigger_time_ms or working.last_checked_ms or working.signal_time_ms)
        if mode == "close_and_hold":
            working = _enter_status(working, WAITING_HOLD_CONFIRMATION, time_ms=anchor_ms, reason="breakout_confirmed")
        elif mode == "close_and_pullback":
            expires_at = anchor_ms + int(working.pullback_expire_minutes) * MINUTE_MS
            working = _enter_status(working, WAITING_PULLBACK, time_ms=anchor_ms, reason="breakout_confirmed")
            working = replace(working, wait_pullback=True, pullback_expires_at_ms=expires_at)
        # close_only / legacy will try to enter directly below.

    # If we are no longer in a pure "pending" state, hand off to the dedicated handlers.
    if working.status == WAITING_HOLD_CONFIRMATION:
        return _update_hold_confirmation_trade(
            working,
            candles,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
            wait_pullback_if_chased=wait_pullback_if_chased,
        )
    if working.status == WAITING_PULLBACK:
        return _update_wait_pullback_trade(
            working,
            candles,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            pullback_tolerance_r=trade.pullback_tolerance_r,
        )

    # Step 1) Look for the breakout confirmation candle (or touch in legacy mode).
    breakout_index: int | None = None
    breakout_candle: Candle | None = None
    for index, candle in enumerate(candles):
        if candle.close_time <= trade.last_checked_ms or candle.close_time <= trade.signal_time_ms:
            continue
        if candle.close_time > trade.expires_at_ms:
            break
        last_checked = max(last_checked, candle.close_time)
        if cancel_if_stop_before_trigger and _pending_should_cancel(trade, candle):
            cancelled = _enter_status(trade, CANCELLED, time_ms=candle.close_time, reason="stop_hit_before_breakout")
            return replace(
                cancelled,
                outcome="触发前止损/失效取消",
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )
        if _is_breakout_confirmed(trade, candle, mode=mode):
            breakout_index = index
            breakout_candle = candle
            break

    if breakout_candle is None or breakout_index is None:
        if now_time_ms >= trade.expires_at_ms:
            expired = _enter_status(trade, EXPIRED, time_ms=trade.expires_at_ms, reason="pending_expired")
            return replace(
                expired,
                outcome="未触发过期",
                exit_time_ms=trade.expires_at_ms,
                last_checked_ms=max(last_checked, trade.expires_at_ms),
            )
        return replace(trade, last_checked_ms=last_checked)

    # Record breakout confirmation.
    breakout_event = "breakout_touch" if mode == "legacy" else "breakout_close"
    confirmed = _enter_status(trade, BREAKOUT_CONFIRMED, time_ms=breakout_candle.close_time, reason=breakout_event)
    confirmed = replace(
        confirmed,
        trigger_time_ms=breakout_candle.close_time,
        trigger_fill_price=breakout_candle.close,
        trigger_confirmed=False,
        hold_confirmation_candles_seen=0,
        pullback_expires_at_ms=None,
        last_checked_ms=max(last_checked, breakout_candle.close_time),
        last_price=breakout_candle.close,
        unrealized_pnl_pct=0.0,
    )

    # Step 2) Mode-specific confirmation.
    if mode == "close_and_hold":
        entered = _enter_status(
            confirmed,
            WAITING_HOLD_CONFIRMATION,
            time_ms=breakout_candle.close_time,
            reason="waiting_hold_confirmation",
        )
        # Continue processing subsequent candles in this same sync batch.
        return _update_hold_confirmation_trade(
            entered,
            candles[breakout_index + 1 :],
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
            wait_pullback_if_chased=wait_pullback_if_chased,
        )

    if mode == "close_and_pullback":
        expires_at = breakout_candle.close_time + int(confirmed.pullback_expire_minutes) * MINUTE_MS
        entered = _enter_status(
            confirmed,
            WAITING_PULLBACK,
            time_ms=breakout_candle.close_time,
            reason="waiting_pullback",
        )
        entered = replace(entered, wait_pullback=True, pullback_expires_at_ms=expires_at)
        return _update_wait_pullback_trade(
            entered,
            candles[breakout_index + 1 :],
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            cancel_if_stop_before_trigger=cancel_if_stop_before_trigger,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            max_entry_slippage_r=max_entry_slippage_r,
            pullback_tolerance_r=confirmed.pullback_tolerance_r,
        )

    # close_only (default) and legacy: enter on breakout confirmation candle close (gated by no-chase).
    decision = _apply_no_chase_rule(
        confirmed,
        breakout_candle,
        candles[: breakout_index + 1],
        max_entry_slippage_r=max_entry_slippage_r,
        max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
    )
    if decision.action == "open":
        opened = _open_trade_from_candle(
            replace(
                confirmed,
                no_chase_passed=True,
                entry_slippage_r=decision.entry_slippage_r,
                trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
                wait_pullback=False,
                skip_reason="",
            ),
            breakout_candle,
            preserve_trigger_info=True,
        )
        return _advance_open_trade(
            opened,
            candles[breakout_index + 1 :],
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
        )
    if decision.action == "wait_pullback" and wait_pullback_if_chased:
        expires_at = breakout_candle.close_time + int(confirmed.pullback_expire_minutes) * MINUTE_MS
        waited = _enter_status(
            confirmed,
            WAITING_PULLBACK,
            time_ms=breakout_candle.close_time,
            reason="no_chase_wait_pullback",
        )
        return replace(
            waited,
            wait_pullback=True,
            pullback_expires_at_ms=expires_at,
            no_chase_passed=False,
            entry_slippage_r=decision.entry_slippage_r,
            trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
            skip_reason=decision.reason,
            last_checked_ms=max(last_checked, breakout_candle.close_time),
            last_price=breakout_candle.close,
            unrealized_pnl_pct=0.0,
        )
    skipped = _enter_status(
        confirmed,
        SKIPPED,
        time_ms=breakout_candle.close_time,
        reason="no_chase_skipped",
    )
    return replace(
        skipped,
        outcome="追单过远/大波动跳过",
        exit_time_ms=breakout_candle.close_time,
        exit_price=breakout_candle.close,
        wait_pullback=False,
        no_chase_passed=False,
        entry_slippage_r=decision.entry_slippage_r,
        trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
        skip_reason=decision.reason,
        last_checked_ms=max(last_checked, breakout_candle.close_time),
        last_price=breakout_candle.close,
        unrealized_pnl_pct=0.0,
    )


def _update_hold_confirmation_trade(
    trade: PaperTrade,
    candles: Sequence[Candle],
    *,
    now_time_ms: int,
    fee_rate: float,
    funding_rate_8h: float,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    cancel_if_stop_before_trigger: bool,
    move_stop_to_breakeven_after_tp1: bool,
    max_entry_slippage_r: float,
    max_trigger_candle_atr_multiple: float,
    wait_pullback_if_chased: bool,
) -> PaperTrade:
    """
    close_and_hold confirmation:
    after breakout is confirmed, require N subsequent candle closes to hold beyond trigger_price.
    """
    last_checked = trade.last_checked_ms
    seen = int(trade.hold_confirmation_candles_seen or 0)
    required = max(1, int(trade.hold_confirmation_candles_required or 1))

    for index, candle in enumerate(candles):
        if candle.close_time <= trade.last_checked_ms or candle.close_time <= trade.signal_time_ms:
            continue
        if candle.close_time > trade.expires_at_ms:
            break
        last_checked = max(last_checked, candle.close_time)

        if cancel_if_stop_before_trigger and _pending_should_cancel(trade, candle):
            cancelled = _enter_status(trade, CANCELLED, time_ms=candle.close_time, reason="stop_hit_during_hold_confirmation")
            return replace(
                cancelled,
                outcome="确认失败：止损先到/反向失效",
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )

        # Hold rule: the next candle(s) must not close back across trigger_price.
        if trade.direction == "做多观察" and candle.close < trade.trigger_price:
            cancelled = _enter_status(
                trade,
                CANCELLED,
                time_ms=candle.close_time,
                reason="hold_failed_close_back_below_trigger",
            )
            return replace(
                cancelled,
                outcome="确认失败：突破后收回触发价下方",
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )
        if trade.direction == "做空观察" and candle.close > trade.trigger_price:
            cancelled = _enter_status(
                trade,
                CANCELLED,
                time_ms=candle.close_time,
                reason="hold_failed_close_back_above_trigger",
            )
            return replace(
                cancelled,
                outcome="确认失败：跌破后收回触发价上方",
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )

        seen += 1
        progressed = replace(trade, hold_confirmation_candles_seen=seen, last_checked_ms=last_checked, last_price=candle.close)
        if seen < required:
            trade = progressed
            continue

        # Hold confirmed: attempt entry at this candle close (still gated by no-chase).
        decision = _apply_no_chase_rule(
            progressed,
            candle,
            candles[: index + 1],
            max_entry_slippage_r=max_entry_slippage_r,
            max_trigger_candle_atr_multiple=max_trigger_candle_atr_multiple,
        )
        if decision.action == "open":
            opened = _open_trade_from_candle(
                replace(
                    progressed,
                    no_chase_passed=True,
                    entry_slippage_r=decision.entry_slippage_r,
                    trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
                    wait_pullback=False,
                    skip_reason="",
                ),
                candle,
                preserve_trigger_info=True,
            )
            return _advance_open_trade(
                opened,
                candles[index + 1 :],
                now_time_ms=now_time_ms,
                fee_rate=fee_rate,
                funding_rate_8h=funding_rate_8h,
                time_stop_minutes=time_stop_minutes,
                min_progress_r=min_progress_r,
                r_trailing_enabled=r_trailing_enabled,
                trailing_trigger_pct=trailing_trigger_pct,
                trailing_lock_pct=trailing_lock_pct,
                move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
            )
        if decision.action == "wait_pullback" and wait_pullback_if_chased:
            expires_at = candle.close_time + int(progressed.pullback_expire_minutes) * MINUTE_MS
            waited = _enter_status(progressed, WAITING_PULLBACK, time_ms=candle.close_time, reason="no_chase_after_hold")
            return replace(
                waited,
                wait_pullback=True,
                pullback_expires_at_ms=expires_at,
                no_chase_passed=False,
                entry_slippage_r=decision.entry_slippage_r,
                trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
                skip_reason=decision.reason,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )

        skipped = _enter_status(progressed, SKIPPED, time_ms=candle.close_time, reason="no_chase_after_hold_skipped")
        return replace(
            skipped,
            outcome="追单过远/大波动跳过",
            exit_time_ms=candle.close_time,
            exit_price=candle.close,
            wait_pullback=False,
            no_chase_passed=False,
            entry_slippage_r=decision.entry_slippage_r,
            trigger_candle_atr_multiple=decision.trigger_candle_atr_multiple,
            skip_reason=decision.reason,
            last_checked_ms=max(last_checked, candle.close_time),
            last_price=candle.close,
            unrealized_pnl_pct=0.0,
        )

    if now_time_ms >= trade.expires_at_ms:
        expired = _enter_status(trade, EXPIRED, time_ms=trade.expires_at_ms, reason="hold_confirmation_expired")
        return replace(
            expired,
            outcome="确认失败：等待持稳过期",
            exit_time_ms=trade.expires_at_ms,
            last_checked_ms=max(last_checked, trade.expires_at_ms),
        )
    return replace(trade, last_checked_ms=last_checked, hold_confirmation_candles_seen=seen)


def _confirmation_candle_quality_ok(direction: str, candle: Candle) -> bool:
    rng = candle.high - candle.low
    if rng <= 0:
        return True
    body_ratio = abs(candle.close - candle.open) / rng
    if body_ratio < MIN_CONFIRM_BODY_RATIO:
        return False
    if direction == "做多观察":
        close_pos = (candle.close - candle.low) / rng
    else:
        close_pos = (candle.high - candle.close) / rng
    return close_pos >= MIN_CONFIRM_CLOSE_POS


def _is_confirmation_candle(trade: PaperTrade, candle: Candle) -> bool:
    # Trigger rule:
    # - Long triggers when 5m candle CLOSE >= trigger_price
    # - Short triggers when 5m candle CLOSE <= trigger_price
    if trade.direction == "做多观察":
        return (
            candle.close >= trade.trigger_price
            and candle.close >= candle.open
            and _confirmation_candle_quality_ok(trade.direction, candle)
        )
    if trade.direction == "做空观察":
        return (
            candle.close <= trade.trigger_price
            and candle.close <= candle.open
            and _confirmation_candle_quality_ok(trade.direction, candle)
        )
    return False


def _is_breakout_confirmed(trade: PaperTrade, candle: Candle, *, mode: str) -> bool:
    """
    Breakout confirmation check.

    - legacy: treat the trigger as "touched" intrabar (high/low), for A/B comparisons
    - others: use close-based confirmation (non-repainting)
    """
    if (mode or "").strip().lower() == "legacy":
        if trade.direction == "做多观察":
            return candle.high >= trade.trigger_price
        if trade.direction == "做空观察":
            return candle.low <= trade.trigger_price
        return False
    return _is_confirmation_candle(trade, candle)


def _pending_should_cancel(trade: PaperTrade, candle: Candle) -> bool:
    """
    Cancel a pending signal before it triggers.

    Rule:
    - If price hits the planned stop before entry, we consider the setup invalid and do not enter.
    """
    if trade.direction == "做多观察":
        return candle.low <= trade.stop_loss
    if trade.direction == "做空观察":
        return candle.high >= trade.stop_loss
    return False


def _avg_true_range(candles: Sequence[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    ranges: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        ranges.append(max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close)))
    return sum(ranges) / period


def _risk_r_from_trigger(trade: PaperTrade) -> float:
    # R is defined on trigger vs stop (not entry), to detect "chase" after breakout confirmation.
    if trade.direction == "做多观察":
        return max(trade.trigger_price - trade.stop_loss, 1e-12)
    if trade.direction == "做空观察":
        return max(trade.stop_loss - trade.trigger_price, 1e-12)
    return 1e-12


def _entry_slippage_r(trade: PaperTrade, entry_price: float) -> float:
    r = _risk_r_from_trigger(trade)
    if trade.direction == "做多观察":
        return (entry_price - trade.trigger_price) / r
    if trade.direction == "做空观察":
        return (trade.trigger_price - entry_price) / r
    return 0.0


@dataclass(frozen=True)
class _NoChaseDecision:
    action: str  # "open" / "wait_pullback" / "skip"
    reason: str
    entry_slippage_r: float | None
    trigger_candle_atr_multiple: float | None


def _apply_no_chase_rule(
    trade: PaperTrade,
    trigger_candle: Candle,
    atr_context: Sequence[Candle],
    *,
    max_entry_slippage_r: float,
    max_trigger_candle_atr_multiple: float,
) -> _NoChaseDecision:
    """
    Anti-chase guard for paper trading.
    It only gates actual entry after a trigger candle is confirmed.
    """
    entry_price = trigger_candle.close
    slippage_r = _entry_slippage_r(trade, entry_price)
    fail_reasons: list[str] = []
    if slippage_r > float(max_entry_slippage_r):
        fail_reasons.append(f"entry_slippage_r={slippage_r:.2f} > {float(max_entry_slippage_r):.2f}")

    atr = _avg_true_range(atr_context, 14)
    atr_multiple: float | None = None
    if atr is not None and atr > 0:
        rng = trigger_candle.high - trigger_candle.low
        body = abs(trigger_candle.close - trigger_candle.open)
        atr_multiple = max(rng, body) / atr
        if atr_multiple > float(max_trigger_candle_atr_multiple):
            fail_reasons.append(
                f"trigger_candle_atr_multiple={atr_multiple:.2f} > {float(max_trigger_candle_atr_multiple):.2f}"
            )

    if not fail_reasons:
        return _NoChaseDecision(
            action="open",
            reason="",
            entry_slippage_r=float(slippage_r),
            trigger_candle_atr_multiple=atr_multiple,
        )

    return _NoChaseDecision(
        action="wait_pullback",
        reason="；".join(fail_reasons),
        entry_slippage_r=float(slippage_r),
        trigger_candle_atr_multiple=atr_multiple,
    )


def _pullback_entry_ok(trade: PaperTrade, candle: Candle, *, tolerance_r: float) -> bool:
    """
    Pullback entry check:
    - long: price pulls back into the trigger area (within tolerance), then closes back >= trigger_price
    - short: price bounces into the trigger area, then closes back <= trigger_price
    """
    r = _risk_r_from_trigger(trade)
    tol = max(0.0, float(tolerance_r)) * r
    if trade.direction == "做多观察":
        return candle.low <= (trade.trigger_price + tol) and candle.close >= trade.trigger_price
    if trade.direction == "做空观察":
        return candle.high >= (trade.trigger_price - tol) and candle.close <= trade.trigger_price
    return False


def _update_wait_pullback_trade(
    trade: PaperTrade,
    candles: Sequence[Candle],
    *,
    now_time_ms: int,
    fee_rate: float,
    funding_rate_8h: float,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    cancel_if_stop_before_trigger: bool,
    move_stop_to_breakeven_after_tp1: bool,
    max_entry_slippage_r: float,
    pullback_tolerance_r: float,
) -> PaperTrade:
    last_checked = trade.last_checked_ms
    expire_at = int(trade.pullback_expires_at_ms or trade.expires_at_ms)
    entry_index: int | None = None
    entry_candle: Candle | None = None
    for index, candle in enumerate(candles):
        if candle.close_time <= trade.last_checked_ms or candle.close_time <= trade.signal_time_ms:
            continue
        if candle.close_time > expire_at:
            break
        last_checked = max(last_checked, candle.close_time)
        if cancel_if_stop_before_trigger and _pending_should_cancel(trade, candle):
            cancelled = _enter_status(trade, CANCELLED, time_ms=candle.close_time, reason="stop_hit_before_pullback_entry")
            return replace(
                cancelled,
                outcome="等待回踩期间止损/失效取消",
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                last_checked_ms=max(last_checked, candle.close_time),
                last_price=candle.close,
                unrealized_pnl_pct=0.0,
            )
        if _pullback_entry_ok(trade, candle, tolerance_r=pullback_tolerance_r):
            # Re-check slippage at the pullback entry close to avoid chasing on large bounces.
            slippage_r = _entry_slippage_r(trade, candle.close)
            if slippage_r > float(max_entry_slippage_r):
                continue
            entry_index = index
            entry_candle = candle
            break

    if entry_candle is not None and entry_index is not None:
        opened = _open_trade_from_candle(
            replace(
                trade,
                no_chase_passed=True,
                # Keep the original trigger-candle slippage/ATR multiple for auditing; do not overwrite.
            ),
            entry_candle,
            preserve_trigger_info=True,
        )
        return _advance_open_trade(
            opened,
            candles[entry_index + 1 :],
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
        )

    if now_time_ms >= expire_at:
        expired = _enter_status(trade, EXPIRED, time_ms=expire_at, reason="pullback_expired")
        return replace(
            expired,
            outcome="等待回踩过期",
            exit_time_ms=expire_at,
            last_checked_ms=max(last_checked, expire_at),
        )
    return replace(trade, last_checked_ms=last_checked)
def _open_trade_from_candle_impl(trade: PaperTrade, candle: Candle, *, preserve_trigger_info: bool) -> PaperTrade:
    entry = candle.close
    is_long = trade.direction == "做多观察"
    risk = abs(entry - trade.stop_loss)
    risk = max(risk, 1e-12)
    tp1 = entry + risk * trade.tp1_r if is_long else entry - risk * trade.tp1_r
    tp2 = entry + risk * trade.tp2_r if is_long else entry - risk * trade.tp2_r
    final_target = entry + risk * trade.final_tp_r if is_long else entry - risk * trade.final_tp_r
    risk_pct = (risk / entry) if entry > 0 else 0.0
    trigger_time_ms = trade.trigger_time_ms if preserve_trigger_info and trade.trigger_time_ms is not None else candle.close_time
    trigger_fill_price = trade.trigger_fill_price if preserve_trigger_info and trade.trigger_fill_price > 0 else candle.close
    entered = _enter_status(trade, TRIGGERED, time_ms=candle.close_time, reason="entry_triggered")
    return replace(
        entered,
        trigger_time_ms=trigger_time_ms,
        trigger_fill_price=trigger_fill_price,
        trigger_confirmed=True,
        entry_time_ms=candle.close_time,
        entry_price=entry,
        tp1_price=tp1,
        tp2_price=tp2,
        final_target_price=final_target,
        active_stop_price=trade.stop_loss,
        last_checked_ms=candle.close_time,
        last_price=entry,
        unrealized_pnl_pct=0.0,
        initial_risk_pct=risk_pct,
        max_favorable_price=entry,
        max_adverse_price=entry,
        max_favorable_pnl_pct=0.0,
        max_adverse_pnl_pct=0.0,
        max_favorable_r=0.0,
        max_adverse_r=0.0,
        time_to_mfe_minutes=0.0,
        time_to_mae_minutes=0.0,
    )


def _open_trade_from_candle(trade: PaperTrade, candle: Candle, *, preserve_trigger_info: bool = False) -> PaperTrade:
    return _open_trade_from_candle_impl(trade, candle, preserve_trigger_info=preserve_trigger_info)


def _advance_open_trade(
    trade: PaperTrade,
    candles: Sequence[Candle],
    *,
    now_time_ms: int,
    fee_rate: float,
    funding_rate_8h: float,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    move_stop_to_breakeven_after_tp1: bool,
) -> PaperTrade:
    if trade.entry_time_ms is None or trade.entry_price <= 0:
        return trade

    is_long = trade.direction == "做多观察"
    entry = trade.entry_price
    risk = abs(entry - trade.stop_loss)
    risk = max(risk, 1e-12)
    tp1 = trade.tp1_price or (entry + risk * trade.tp1_r if is_long else entry - risk * trade.tp1_r)
    tp2 = trade.tp2_price or (entry + risk * trade.tp2_r if is_long else entry - risk * trade.tp2_r)
    final_tp = trade.final_target_price or (entry + risk * trade.final_tp_r if is_long else entry - risk * trade.final_tp_r)
    initial_risk_pct = trade.initial_risk_pct if trade.initial_risk_pct > 0 else ((risk / entry) if entry > 0 else 0.0)
    partial_weight = max(0.0, min(1.0, float(trade.partial_close_pct_at_tp1)))
    active_stop = trade.active_stop_price or trade.stop_loss
    trailing_stop = trade.trailing_stop_price
    trailing_active = trade.trailing_stop_activated
    tp1_hit = trade.tp1_hit
    tp1_time = trade.tp1_time_ms or 0
    tp1_hit_price = trade.tp1_hit_price
    tp2_hit = trade.tp2_hit
    tp2_time = trade.tp2_time_ms or 0
    tp2_hit_price = trade.tp2_hit_price
    moved_be = trade.moved_stop_to_breakeven
    best_progress_r = trade.best_progress_r
    last_checked = trade.last_checked_ms
    hold_until_ms = trade.entry_time_ms + int(max(0.25, trade.planned_hold_hours) * HOUR_MS)
    time_stop_at = trade.entry_time_ms + time_stop_minutes * MINUTE_MS

    # MFE/MAE tracking (since entry).
    mfe_price = trade.max_favorable_price or entry
    mae_price = trade.max_adverse_price or entry
    mfe_pnl = trade.max_favorable_pnl_pct
    mae_pnl = trade.max_adverse_pnl_pct
    mfe_r = trade.max_favorable_r
    mae_r = trade.max_adverse_r
    t_mfe = trade.time_to_mfe_minutes
    t_mae = trade.time_to_mae_minutes
    if mfe_price <= 0 or mae_price <= 0:
        mfe_price = entry
        mae_price = entry
        mfe_pnl = 0.0
        mae_pnl = 0.0
        mfe_r = 0.0
        mae_r = 0.0
        t_mfe = 0.0
        t_mae = 0.0

    for candle in candles:
        if candle.close_time <= last_checked:
            continue
        # Update MFE/MAE before any exit checks so the exit candle's range is reflected.
        if is_long:
            if candle.high > mfe_price:
                mfe_price = candle.high
                mfe_pnl = (mfe_price - entry) / entry if entry > 0 else 0.0
                mfe_r = (mfe_price - entry) / risk if risk > 0 else 0.0
                t_mfe = (candle.close_time - trade.entry_time_ms) / MINUTE_MS
            if candle.low < mae_price:
                mae_price = candle.low
                mae_pnl = (mae_price - entry) / entry if entry > 0 else 0.0
                mae_r = (mae_price - entry) / risk if risk > 0 else 0.0
                t_mae = (candle.close_time - trade.entry_time_ms) / MINUTE_MS
        else:
            if candle.low < mfe_price:
                mfe_price = candle.low
                mfe_pnl = (entry - mfe_price) / entry if entry > 0 else 0.0
                mfe_r = (entry - mfe_price) / risk if risk > 0 else 0.0
                t_mfe = (candle.close_time - trade.entry_time_ms) / MINUTE_MS
            if candle.high > mae_price:
                mae_price = candle.high
                mae_pnl = (entry - mae_price) / entry if entry > 0 else 0.0
                mae_r = (entry - mae_price) / risk if risk > 0 else 0.0
                t_mae = (candle.close_time - trade.entry_time_ms) / MINUTE_MS
        if not tp2_hit:
            if is_long and candle.high >= tp2:
                tp2_hit = True
                tp2_time = candle.close_time
                tp2_hit_price = tp2
            elif (not is_long) and candle.low <= tp2:
                tp2_hit = True
                tp2_time = candle.close_time
                tp2_hit_price = tp2
        if is_long:
            best_progress_r = max(best_progress_r, (candle.high - entry) / risk)
            if candle.low <= active_stop:
                outcome = "移动止损" if trailing_active else ("半仓止盈后保本" if tp1_hit else "止损")
                pnl = (
                    _partial_pnl_pct(
                        direction=trade.direction,
                        entry=entry,
                        first_exit=tp1,
                        final_exit=active_stop,
                        fee_rate=fee_rate,
                        first_weight=partial_weight,
                    )
                    if tp1_hit
                    else _pnl_pct(trade.direction, entry, active_stop, fee_rate)
                )
                enriched = replace(
                    trade,
                    max_favorable_price=mfe_price,
                    max_adverse_price=mae_price,
                    max_favorable_pnl_pct=mfe_pnl,
                    max_adverse_pnl_pct=mae_pnl,
                    max_favorable_r=mfe_r,
                    max_adverse_r=mae_r,
                    time_to_mfe_minutes=t_mfe,
                    time_to_mae_minutes=t_mae,
                    tp2_price=tp2,
                    final_target_price=final_tp,
                    tp1_hit_price=tp1_hit_price,
                    tp2_hit=tp2_hit,
                    tp2_time_ms=tp2_time or None,
                    tp2_hit_price=tp2_hit_price,
                    moved_stop_to_breakeven=moved_be,
                    initial_risk_pct=initial_risk_pct,
                )
                return _close_trade(
                    enriched,
                    exit_time_ms=candle.close_time,
                    exit_price=active_stop,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=final_tp,
                    active_stop_price=active_stop,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    tp1_hit=tp1_hit,
                    tp1_time_ms=tp1_time or None,
                    best_progress_r=best_progress_r,
                )
            if r_trailing_enabled:
                r_stop = _r_trailing_stop_price(
                    direction=trade.direction,
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
                tp1_hit_price = tp1
                if move_stop_to_breakeven_after_tp1:
                    active_stop = max(active_stop, entry)
                    moved_be = True
                last_checked = candle.close_time
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.high >= final_tp:
                pnl = _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=final_tp,
                    fee_rate=fee_rate,
                    first_weight=partial_weight,
                )
                return _close_trade(
                    replace(
                        trade,
                        max_favorable_price=mfe_price,
                        max_adverse_price=mae_price,
                        max_favorable_pnl_pct=mfe_pnl,
                        max_adverse_pnl_pct=mae_pnl,
                        max_favorable_r=mfe_r,
                        max_adverse_r=mae_r,
                        time_to_mfe_minutes=t_mfe,
                        time_to_mae_minutes=t_mae,
                        tp2_price=tp2,
                        final_target_price=final_tp,
                        tp1_hit_price=tp1_hit_price,
                        tp2_hit=tp2_hit,
                        tp2_time_ms=tp2_time or None,
                        tp2_hit_price=tp2_hit_price,
                        moved_stop_to_breakeven=moved_be,
                        initial_risk_pct=initial_risk_pct,
                    ),
                    exit_time_ms=candle.close_time,
                    exit_price=final_tp,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=final_tp,
                    active_stop_price=active_stop,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    tp1_hit=True,
                    tp1_time_ms=tp1_time or None,
                    best_progress_r=best_progress_r,
                )
        else:
            best_progress_r = max(best_progress_r, (entry - candle.low) / risk)
            if candle.high >= active_stop:
                outcome = "移动止损" if trailing_active else ("半仓止盈后保本" if tp1_hit else "止损")
                pnl = (
                    _partial_pnl_pct(
                        direction=trade.direction,
                        entry=entry,
                        first_exit=tp1,
                        final_exit=active_stop,
                        fee_rate=fee_rate,
                        first_weight=partial_weight,
                    )
                    if tp1_hit
                    else _pnl_pct(trade.direction, entry, active_stop, fee_rate)
                )
                enriched = replace(
                    trade,
                    max_favorable_price=mfe_price,
                    max_adverse_price=mae_price,
                    max_favorable_pnl_pct=mfe_pnl,
                    max_adverse_pnl_pct=mae_pnl,
                    max_favorable_r=mfe_r,
                    max_adverse_r=mae_r,
                    time_to_mfe_minutes=t_mfe,
                    time_to_mae_minutes=t_mae,
                    tp2_price=tp2,
                    final_target_price=final_tp,
                    tp1_hit_price=tp1_hit_price,
                    tp2_hit=tp2_hit,
                    tp2_time_ms=tp2_time or None,
                    tp2_hit_price=tp2_hit_price,
                    moved_stop_to_breakeven=moved_be,
                    initial_risk_pct=initial_risk_pct,
                )
                return _close_trade(
                    enriched,
                    exit_time_ms=candle.close_time,
                    exit_price=active_stop,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=final_tp,
                    active_stop_price=active_stop,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    tp1_hit=tp1_hit,
                    tp1_time_ms=tp1_time or None,
                    best_progress_r=best_progress_r,
                )
            if r_trailing_enabled:
                r_stop = _r_trailing_stop_price(
                    direction=trade.direction,
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
                tp1_hit_price = tp1
                if move_stop_to_breakeven_after_tp1:
                    active_stop = min(active_stop, entry)
                    moved_be = True
                last_checked = candle.close_time
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.low <= final_tp:
                pnl = _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=final_tp,
                    fee_rate=fee_rate,
                    first_weight=partial_weight,
                )
                return _close_trade(
                    replace(
                        trade,
                        max_favorable_price=mfe_price,
                        max_adverse_price=mae_price,
                        max_favorable_pnl_pct=mfe_pnl,
                        max_adverse_pnl_pct=mae_pnl,
                        max_favorable_r=mfe_r,
                        max_adverse_r=mae_r,
                        time_to_mfe_minutes=t_mfe,
                        time_to_mae_minutes=t_mae,
                        tp2_price=tp2,
                        final_target_price=final_tp,
                        tp1_hit_price=tp1_hit_price,
                        tp2_hit=tp2_hit,
                        tp2_time_ms=tp2_time or None,
                        tp2_hit_price=tp2_hit_price,
                        moved_stop_to_breakeven=moved_be,
                        initial_risk_pct=initial_risk_pct,
                    ),
                    exit_time_ms=candle.close_time,
                    exit_price=final_tp,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=final_tp,
                    active_stop_price=active_stop,
                    trailing_stop_price=trailing_stop,
                    trailing_stop_activated=trailing_active,
                    tp1_hit=True,
                    tp1_time_ms=tp1_time or None,
                    best_progress_r=best_progress_r,
                )

        if time_stop_minutes > 0 and candle.close_time >= time_stop_at and best_progress_r < min_progress_r:
            pnl = (
                _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=candle.close,
                    fee_rate=fee_rate,
                    first_weight=partial_weight,
                )
                if tp1_hit
                else _pnl_pct(trade.direction, entry, candle.close, fee_rate)
            )
            enriched = replace(
                trade,
                max_favorable_price=mfe_price,
                max_adverse_price=mae_price,
                max_favorable_pnl_pct=mfe_pnl,
                max_adverse_pnl_pct=mae_pnl,
                max_favorable_r=mfe_r,
                max_adverse_r=mae_r,
                time_to_mfe_minutes=t_mfe,
                time_to_mae_minutes=t_mae,
                tp2_price=tp2,
                final_target_price=final_tp,
                tp1_hit_price=tp1_hit_price,
                tp2_hit=tp2_hit,
                tp2_time_ms=tp2_time or None,
                tp2_hit_price=tp2_hit_price,
                moved_stop_to_breakeven=moved_be,
                initial_risk_pct=initial_risk_pct,
            )
            return _close_trade(
                enriched,
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                outcome="时间止损",
                pnl_pct=pnl,
                funding_rate_8h=funding_rate_8h,
                tp1_price=tp1,
                final_target_price=final_tp,
                active_stop_price=active_stop,
                trailing_stop_price=trailing_stop,
                trailing_stop_activated=trailing_active,
                tp1_hit=tp1_hit,
                tp1_time_ms=tp1_time or None,
                best_progress_r=best_progress_r,
            )

        if candle.close_time >= hold_until_ms:
            pnl = (
                _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=candle.close,
                    fee_rate=fee_rate,
                    first_weight=partial_weight,
                )
                if tp1_hit
                else _pnl_pct(trade.direction, entry, candle.close, fee_rate)
            )
            enriched = replace(
                trade,
                max_favorable_price=mfe_price,
                max_adverse_price=mae_price,
                max_favorable_pnl_pct=mfe_pnl,
                max_adverse_pnl_pct=mae_pnl,
                max_favorable_r=mfe_r,
                max_adverse_r=mae_r,
                time_to_mfe_minutes=t_mfe,
                time_to_mae_minutes=t_mae,
                tp2_price=tp2,
                final_target_price=final_tp,
                tp1_hit_price=tp1_hit_price,
                tp2_hit=tp2_hit,
                tp2_time_ms=tp2_time or None,
                tp2_hit_price=tp2_hit_price,
                moved_stop_to_breakeven=moved_be,
                initial_risk_pct=initial_risk_pct,
            )
            return _close_trade(
                enriched,
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                outcome="半仓止盈后到期" if tp1_hit else "到期平仓",
                pnl_pct=pnl,
                funding_rate_8h=funding_rate_8h,
                tp1_price=tp1,
                final_target_price=final_tp,
                active_stop_price=active_stop,
                trailing_stop_price=trailing_stop,
                trailing_stop_activated=trailing_active,
                tp1_hit=tp1_hit,
                tp1_time_ms=tp1_time or None,
                best_progress_r=best_progress_r,
            )
        last_checked = candle.close_time

    out = replace(
        trade,
        tp1_price=tp1,
        tp2_price=tp2,
        final_target_price=final_tp,
        active_stop_price=active_stop,
        tp1_hit=tp1_hit,
        tp1_time_ms=tp1_time or None,
        tp1_hit_price=tp1_hit_price,
        tp2_hit=tp2_hit,
        tp2_time_ms=tp2_time or None,
        tp2_hit_price=tp2_hit_price,
        moved_stop_to_breakeven=moved_be,
        initial_risk_pct=initial_risk_pct,
        trailing_stop_price=trailing_stop,
        trailing_stop_activated=trailing_active,
        best_progress_r=best_progress_r,
        last_checked_ms=last_checked,
        max_favorable_price=mfe_price,
        max_adverse_price=mae_price,
        max_favorable_pnl_pct=mfe_pnl,
        max_adverse_pnl_pct=mae_pnl,
        max_favorable_r=mfe_r,
        max_adverse_r=mae_r,
        time_to_mfe_minutes=t_mfe,
        time_to_mae_minutes=t_mae,
    )
    return _mark_to_market(out, candles)


def _mark_to_market(trade: PaperTrade, candles: Sequence[Candle]) -> PaperTrade:
    if not candles:
        return trade
    last = candles[-1].close
    if trade.status not in {OPEN, TRIGGERED} or trade.entry_price <= 0:
        return replace(trade, last_price=last, unrealized_pnl_pct=0.0)
    gross = (last - trade.entry_price) / trade.entry_price
    if trade.direction == "做空观察":
        gross = (trade.entry_price - last) / trade.entry_price
    return replace(trade, last_price=last, unrealized_pnl_pct=gross)


def _close_trade(
    trade: PaperTrade,
    *,
    exit_time_ms: int,
    exit_price: float,
    outcome: str,
    pnl_pct: float,
    funding_rate_8h: float,
    tp1_price: float,
    final_target_price: float,
    active_stop_price: float,
    trailing_stop_price: float,
    trailing_stop_activated: bool,
    tp1_hit: bool,
    tp1_time_ms: int | None,
    best_progress_r: float,
) -> PaperTrade:
    funding_cost = _funding_cost_pct(trade.entry_time_ms or exit_time_ms, exit_time_ms, funding_rate_8h)
    net_pnl_pct = pnl_pct - funding_cost
    risk_pct = trade.initial_risk_pct
    if risk_pct <= 0 and trade.entry_price > 0 and trade.stop_loss > 0:
        risk = abs(trade.entry_price - trade.stop_loss)
        if risk > 0:
            risk_pct = risk / trade.entry_price
    realized_r = (net_pnl_pct / risk_pct) if risk_pct > 0 else 0.0
    return replace(
        trade,
        status=CLOSED,
        exit_time_ms=exit_time_ms,
        exit_price=exit_price,
        outcome=outcome,
        pnl_pct=net_pnl_pct,
        funding_cost_pct=funding_cost,
        initial_risk_pct=risk_pct,
        realized_r_multiple=realized_r,
        tp1_price=tp1_price,
        final_target_price=final_target_price,
        active_stop_price=active_stop_price,
        trailing_stop_price=trailing_stop_price,
        trailing_stop_activated=trailing_stop_activated,
        tp1_hit=tp1_hit,
        tp1_time_ms=tp1_time_ms,
        best_progress_r=best_progress_r,
        last_checked_ms=exit_time_ms,
    )


def _paper_trade_line_zh(trade: PaperTrade) -> str:
    status = {
        PENDING: "待触发",
        TRIGGERED: "已触发",
        OPEN: "持仓中",
        CLOSED: "已平仓",
        EXPIRED: "未触发过期",
        CANCELLED: "已取消",
    }.get(trade.status, trade.status)
    if trade.status == PENDING:
        return f"- {trade.symbol} {trade.direction}｜{status}｜触发价 {trade.trigger_price:g}｜过期 {_format_ms(trade.expires_at_ms)}"
    if trade.status in {TRIGGERED, OPEN}:
        active_stop = trade.active_stop_price or trade.stop_loss
        return (
            f"- {trade.symbol} {trade.direction}｜{status}｜入场 {trade.entry_price:g}｜"
            f"当前保护止损 {active_stop:g}｜目标 {trade.final_target_price:g}"
        )
    return (
        f"- {trade.symbol} {trade.direction}｜{status}｜{trade.outcome or '-'}｜"
        f"收益 {trade.pnl_pct * 100:.2f}%｜出场 {trade.exit_price:g}"
    )


def _format_ms(value: int | None) -> str:
    if value is None:
        return "-"
    return datetime.fromtimestamp(value / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
