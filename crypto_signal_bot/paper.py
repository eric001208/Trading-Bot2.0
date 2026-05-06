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
OPEN = "open"
CLOSED = "closed"
EXPIRED = "expired"
ACTIVE_STATUSES = {PENDING, OPEN}

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS


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
    entry_time_ms: int | None = None
    entry_price: float = 0.0
    exit_time_ms: int | None = None
    exit_price: float = 0.0
    tp1_price: float = 0.0
    final_target_price: float = 0.0
    active_stop_price: float = 0.0
    tp1_hit: bool = False
    tp1_time_ms: int | None = None
    trailing_stop_price: float = 0.0
    trailing_stop_activated: bool = False
    best_progress_r: float = 0.0
    outcome: str = ""
    pnl_pct: float = 0.0
    funding_cost_pct: float = 0.0
    last_checked_ms: int = 0


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
    dedupe_minutes: int = 60,
) -> list[PaperTrade]:
    signal_at = signal_time_ms if signal_time_ms is not None else now_ms()
    trades = load_paper_trades(path)
    recorded: list[PaperTrade] = []
    for candidate in candidates:
        if candidate.direction not in {"做多观察", "做空观察"}:
            continue
        if _has_duplicate_active_trade(trades, candidate, signal_at, dedupe_minutes):
            continue
        trade = _paper_trade_from_candidate(
            candidate,
            signal_time_ms=signal_at,
            confirm_minutes=confirm_minutes,
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
    dedupe_minutes: int = 60,
) -> PaperTrade | None:
    recorded = record_signal_candidates(
        [candidate],
        path,
        signal_time_ms=signal_time_ms,
        confirm_minutes=confirm_minutes,
        dedupe_minutes=dedupe_minutes,
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
) -> PaperTrade:
    if trade.status not in ACTIVE_STATUSES:
        return trade

    ordered = sorted((c for c in candles if c.is_closed), key=lambda c: c.close_time)
    if trade.status == PENDING:
        return _update_pending_trade(
            trade,
            ordered,
            now_time_ms=now_time_ms,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
        )
    return _advance_open_trade(
        trade,
        ordered,
        now_time_ms=now_time_ms,
        fee_rate=fee_rate,
        funding_rate_8h=funding_rate_8h,
        time_stop_minutes=time_stop_minutes,
        min_progress_r=min_progress_r,
        r_trailing_enabled=r_trailing_enabled,
        trailing_trigger_pct=trailing_trigger_pct,
        trailing_lock_pct=trailing_lock_pct,
    )


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
    open_count = sum(1 for trade in trades if trade.status == OPEN)
    expired = sum(1 for trade in trades if trade.status == EXPIRED)
    closed = [trade for trade in trades if trade.status == CLOSED]
    wins = sum(1 for trade in closed if trade.pnl_pct > 0)
    losses = sum(1 for trade in closed if trade.pnl_pct <= 0)
    total_return = sum(trade.pnl_pct for trade in closed) * 100
    win_rate = wins / len(closed) * 100 if closed else 0.0
    funding_cost = sum(trade.funding_cost_pct for trade in closed) * 100

    lines = [
        "虚拟盘记录摘要",
        f"总记录：{total} 笔",
        f"待触发：{pending} 笔，持仓中：{open_count} 笔，已平仓：{len(closed)} 笔，未触发过期：{expired} 笔",
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


def _paper_trade_from_dict(item: Any) -> PaperTrade:
    if not isinstance(item, dict):
        raise ValueError("虚拟盘账本存在非法记录")
    names = {field.name for field in fields(PaperTrade)}
    values = {name: item[name] for name in names if name in item}
    if "reasons" in values:
        values["reasons"] = tuple(values["reasons"])
    return PaperTrade(**values)


def _paper_trade_from_candidate(
    candidate: ObservationCandidate,
    *,
    signal_time_ms: int,
    confirm_minutes: int,
) -> PaperTrade:
    confirm_until_ms = signal_time_ms + max(1, confirm_minutes) * MINUTE_MS
    expires_at_ms = signal_time_ms + max(1, candidate.expires_after_minutes) * MINUTE_MS
    return PaperTrade(
        paper_id=_candidate_id(candidate, signal_time_ms),
        symbol=candidate.symbol.strip().upper(),
        direction=candidate.direction,
        score=candidate.score,
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
    )


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


def _fetch_start_ms(trade: PaperTrade) -> int:
    anchor = trade.last_checked_ms or trade.signal_time_ms
    return max(0, anchor - 2 * MINUTE_MS)


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
) -> PaperTrade:
    last_checked = trade.last_checked_ms
    entry_index: int | None = None
    entry_candle: Candle | None = None
    for index, candle in enumerate(candles):
        if candle.close_time <= trade.last_checked_ms or candle.close_time <= trade.signal_time_ms:
            continue
        if candle.close_time > trade.confirm_until_ms:
            break
        last_checked = max(last_checked, candle.close_time)
        if _is_confirmation_candle(trade, candle):
            entry_index = index
            entry_candle = candle
            break

    if entry_candle is not None and entry_index is not None:
        opened = _open_trade_from_candle(trade, entry_candle)
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
        )

    if now_time_ms >= trade.expires_at_ms:
        return replace(
            trade,
            status=EXPIRED,
            outcome="未触发过期",
            exit_time_ms=trade.expires_at_ms,
            last_checked_ms=max(last_checked, trade.expires_at_ms),
        )
    return replace(trade, last_checked_ms=last_checked)


def _is_confirmation_candle(trade: PaperTrade, candle: Candle) -> bool:
    if trade.direction == "做多观察":
        return candle.close >= trade.trigger_price and candle.close >= candle.open
    if trade.direction == "做空观察":
        return candle.close <= trade.trigger_price and candle.close <= candle.open
    return False


def _open_trade_from_candle(trade: PaperTrade, candle: Candle) -> PaperTrade:
    entry = candle.close
    is_long = trade.direction == "做多观察"
    risk = max(abs(entry - trade.stop_loss), entry * 0.001)
    tp1 = entry + risk if is_long else entry - risk
    final_target = entry + trade.target_rr * risk if is_long else entry - trade.target_rr * risk
    return replace(
        trade,
        status=OPEN,
        entry_time_ms=candle.close_time,
        entry_price=entry,
        tp1_price=tp1,
        final_target_price=final_target,
        active_stop_price=trade.stop_loss,
        last_checked_ms=candle.close_time,
    )


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
) -> PaperTrade:
    if trade.entry_time_ms is None or trade.entry_price <= 0:
        return trade

    is_long = trade.direction == "做多观察"
    entry = trade.entry_price
    risk = max(abs(entry - trade.stop_loss), entry * 0.001)
    tp1 = trade.tp1_price or (entry + risk if is_long else entry - risk)
    tp2 = trade.final_target_price or (entry + trade.target_rr * risk if is_long else entry - trade.target_rr * risk)
    active_stop = trade.active_stop_price or trade.stop_loss
    trailing_stop = trade.trailing_stop_price
    trailing_active = trade.trailing_stop_activated
    tp1_hit = trade.tp1_hit
    tp1_time = trade.tp1_time_ms or 0
    best_progress_r = trade.best_progress_r
    last_checked = trade.last_checked_ms
    hold_until_ms = trade.entry_time_ms + int(max(0.25, trade.planned_hold_hours) * HOUR_MS)
    time_stop_at = trade.entry_time_ms + time_stop_minutes * MINUTE_MS

    for candle in candles:
        if candle.close_time <= last_checked:
            continue
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
                    )
                    if tp1_hit
                    else _pnl_pct(trade.direction, entry, active_stop, fee_rate)
                )
                return _close_trade(
                    trade,
                    exit_time_ms=candle.close_time,
                    exit_price=active_stop,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=tp2,
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
                active_stop = max(active_stop, entry)
                last_checked = candle.close_time
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.high >= tp2:
                pnl = _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=tp2,
                    fee_rate=fee_rate,
                )
                return _close_trade(
                    trade,
                    exit_time_ms=candle.close_time,
                    exit_price=tp2,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=tp2,
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
                    )
                    if tp1_hit
                    else _pnl_pct(trade.direction, entry, active_stop, fee_rate)
                )
                return _close_trade(
                    trade,
                    exit_time_ms=candle.close_time,
                    exit_price=active_stop,
                    outcome=outcome,
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=tp2,
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
                active_stop = min(active_stop, entry)
                last_checked = candle.close_time
                continue
            if tp1_hit and candle.close_time > tp1_time and candle.low <= tp2:
                pnl = _partial_pnl_pct(
                    direction=trade.direction,
                    entry=entry,
                    first_exit=tp1,
                    final_exit=tp2,
                    fee_rate=fee_rate,
                )
                return _close_trade(
                    trade,
                    exit_time_ms=candle.close_time,
                    exit_price=tp2,
                    outcome="分批止盈",
                    pnl_pct=pnl,
                    funding_rate_8h=funding_rate_8h,
                    tp1_price=tp1,
                    final_target_price=tp2,
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
                )
                if tp1_hit
                else _pnl_pct(trade.direction, entry, candle.close, fee_rate)
            )
            return _close_trade(
                trade,
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                outcome="时间止损",
                pnl_pct=pnl,
                funding_rate_8h=funding_rate_8h,
                tp1_price=tp1,
                final_target_price=tp2,
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
                )
                if tp1_hit
                else _pnl_pct(trade.direction, entry, candle.close, fee_rate)
            )
            return _close_trade(
                trade,
                exit_time_ms=candle.close_time,
                exit_price=candle.close,
                outcome="半仓止盈后到期" if tp1_hit else "到期平仓",
                pnl_pct=pnl,
                funding_rate_8h=funding_rate_8h,
                tp1_price=tp1,
                final_target_price=tp2,
                active_stop_price=active_stop,
                trailing_stop_price=trailing_stop,
                trailing_stop_activated=trailing_active,
                tp1_hit=tp1_hit,
                tp1_time_ms=tp1_time or None,
                best_progress_r=best_progress_r,
            )
        last_checked = candle.close_time

    return replace(
        trade,
        tp1_price=tp1,
        final_target_price=tp2,
        active_stop_price=active_stop,
        tp1_hit=tp1_hit,
        tp1_time_ms=tp1_time or None,
        trailing_stop_price=trailing_stop,
        trailing_stop_activated=trailing_active,
        best_progress_r=best_progress_r,
        last_checked_ms=last_checked,
    )


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
    return replace(
        trade,
        status=CLOSED,
        exit_time_ms=exit_time_ms,
        exit_price=exit_price,
        outcome=outcome,
        pnl_pct=pnl_pct - funding_cost,
        funding_cost_pct=funding_cost,
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
        OPEN: "持仓中",
        CLOSED: "已平仓",
        EXPIRED: "未触发过期",
    }.get(trade.status, trade.status)
    if trade.status == PENDING:
        return f"- {trade.symbol} {trade.direction}｜{status}｜触发价 {trade.trigger_price:g}｜过期 {_format_ms(trade.expires_at_ms)}"
    if trade.status == OPEN:
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
