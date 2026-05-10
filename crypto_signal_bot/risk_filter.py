from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

from crypto_signal_bot.paper import ACTIVE_STATUSES, PaperTrade
from crypto_signal_bot.strategies import ObservationCandidate

# If a correlated signal is "much stronger" than the current one, we may allow it through.
# Score is 0-100, so +5 is already a meaningful jump.
CORRELATED_OVERRIDE_SCORE_GAP = 5


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _side_from_direction(direction: str) -> str:
    d = (direction or "").strip().lower()
    if "做多" in direction or d == "long":
        return "long"
    if "做空" in direction or d == "short":
        return "short"
    # Fallback: keep original tag for visibility.
    return d or "unknown"


def _trade_event_time_ms(trade: PaperTrade) -> int:
    # For pending signals we only have signal_time_ms; for open trades we have entry_time_ms.
    return int(trade.entry_time_ms or trade.signal_time_ms)


def _is_stop_like_outcome(outcome: str) -> bool:
    return bool(outcome) and ("止损" in outcome)


@dataclass(frozen=True)
class RiskFilterConfig:
    max_open_trades: int = 1
    max_open_trades_per_side: int = 1
    symbol_cooldown_minutes: int = 60
    side_cooldown_minutes: int = 45
    main_symbol_allowlist: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    altcoin_enabled: bool = False


@dataclass(frozen=True)
class SkippedSignal:
    time_ms: int
    symbol: str
    side: str
    score: int
    skip_reason: str


@dataclass
class _ActiveSlot:
    symbol: str
    side: str
    score: int
    time_ms: int


def filter_signals(
    candidates: Sequence[ObservationCandidate],
    trades: Sequence[PaperTrade],
    *,
    now_time_ms: int,
    config: RiskFilterConfig,
) -> tuple[list[ObservationCandidate], list[SkippedSignal]]:
    """
    Filter qualified observation signals with risk/correlation/cooldown rules.

    We treat ACTIVE_STATUSES (pending/triggered/open) as "active exposure" to avoid
    recording multiple highly correlated signals that can trigger at the same time.
    """

    allowlist = {_norm_symbol(s) for s in config.main_symbol_allowlist if s.strip()}

    active_slots: list[_ActiveSlot] = []
    active_symbol_side: set[tuple[str, str]] = set()
    for t in trades:
        if t.status not in ACTIVE_STATUSES:
            continue
        sym = _norm_symbol(t.symbol)
        side = _side_from_direction(t.direction)
        active_slots.append(_ActiveSlot(sym, side, int(t.score), _trade_event_time_ms(t)))
        active_symbol_side.add((sym, side))

    # Latest "signal/entry time" per side (used for side cooldown).
    # We consider all historical trades here (not only active ones) to avoid rapid-fire entries.
    last_side_time: dict[str, int] = {}
    for t in trades:
        side = _side_from_direction(t.direction)
        if side == "unknown":
            continue
        last_side_time[side] = max(last_side_time.get(side, 0), _trade_event_time_ms(t))

    # Latest stop-like exit time per (symbol, side) for symbol cooldown.
    last_stop_exit: dict[tuple[str, str], int] = {}
    for t in trades:
        if not _is_stop_like_outcome(t.outcome or ""):
            continue
        if t.exit_time_ms is None:
            continue
        sym = _norm_symbol(t.symbol)
        side = _side_from_direction(t.direction)
        key = (sym, side)
        last_stop_exit[key] = max(last_stop_exit.get(key, 0), int(t.exit_time_ms))

    accepted: list[ObservationCandidate] = []
    skipped: list[SkippedSignal] = []

    def _active_count() -> int:
        return len(active_slots)

    def _active_count_side(side: str) -> int:
        return sum(1 for slot in active_slots if slot.side == side)

    def _skip(candidate: ObservationCandidate, reason: str) -> None:
        skipped.append(
            SkippedSignal(
                time_ms=int(now_time_ms),
                symbol=_norm_symbol(candidate.symbol),
                side=_side_from_direction(candidate.direction),
                score=int(candidate.score),
                skip_reason=reason,
            )
        )

    for cand in sorted(candidates, key=lambda c: int(c.score), reverse=True):
        sym = _norm_symbol(cand.symbol)
        side = _side_from_direction(cand.direction)

        if not config.altcoin_enabled and allowlist and sym not in allowlist:
            _skip(cand, "altcoin_disabled")
            continue

        if (sym, side) in active_symbol_side:
            _skip(cand, "symbol_side_already_active")
            continue

        if config.symbol_cooldown_minutes > 0:
            key = (sym, side)
            last_exit = last_stop_exit.get(key)
            if last_exit is not None:
                cooldown_ms = int(config.symbol_cooldown_minutes) * 60_000
                if now_time_ms - last_exit < cooldown_ms:
                    mins_left = max(0, int((cooldown_ms - (now_time_ms - last_exit)) / 60_000))
                    _skip(cand, f"symbol_cooldown_after_stop({mins_left}m)")
                    continue

        if config.max_open_trades > 0 and _active_count() >= int(config.max_open_trades):
            _skip(cand, "max_open_trades")
            continue

        if config.max_open_trades_per_side > 0 and _active_count_side(side) >= int(config.max_open_trades_per_side):
            _skip(cand, "max_open_trades_per_side")
            continue

        if config.side_cooldown_minutes > 0:
            last_time = last_side_time.get(side)
            if last_time is not None and last_time > 0:
                cooldown_ms = int(config.side_cooldown_minutes) * 60_000
                if now_time_ms - last_time < cooldown_ms:
                    mins_left = max(0, int((cooldown_ms - (now_time_ms - last_time)) / 60_000))
                    _skip(cand, f"side_cooldown({mins_left}m)")
                    continue

        # Correlation filter: treat main symbols as one correlated bucket per side.
        if allowlist and sym in allowlist:
            correlated = [slot for slot in active_slots if slot.side == side and slot.symbol in allowlist]
            correlated = [slot for slot in correlated if slot.symbol != sym]
            if correlated:
                anchor = next((slot for slot in correlated if slot.symbol == "BTCUSDT"), None) or max(
                    correlated, key=lambda s: s.score
                )
                if int(cand.score) < int(anchor.score) + CORRELATED_OVERRIDE_SCORE_GAP:
                    _skip(cand, f"correlated_with_{anchor.symbol}(score={anchor.score})")
                    continue

        accepted.append(cand)
        active_slots.append(_ActiveSlot(sym, side, int(cand.score), int(now_time_ms)))
        active_symbol_side.add((sym, side))
        last_side_time[side] = int(now_time_ms)

    return accepted, skipped


def append_skipped_signals(path: str | Path, records: Sequence[SkippedSignal]) -> Path:
    out = Path(path)
    if not records:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["time", "symbol", "side", "score", "skip_reason"],
        )
        if not exists:
            w.writeheader()
        for r in records:
            dt = datetime.fromtimestamp(r.time_ms / 1000, tz=UTC)
            w.writerow(
                {
                    "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": r.symbol,
                    "side": r.side,
                    "score": r.score,
                    "skip_reason": r.skip_reason,
                }
            )
    return out
