from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import httpx

from crypto_signal_bot.alerts import TelegramNotifier
from crypto_signal_bot.backtest import BacktestSummary, export_backtest_trades_csv, run_observation_backtest
from crypto_signal_bot.config import Settings
from crypto_signal_bot.indicators import closes, ema, ema_slope
from crypto_signal_bot.logging_setup import configure_logging
from crypto_signal_bot.market import fetch_top_usdm_symbols_by_quote_volume, fetch_usdm_klines, fetch_usdm_klines_range
from crypto_signal_bot.models import Candle
from crypto_signal_bot.paper import (
    CLOSED,
    OPEN,
    PENDING,
    PaperTrade,
    export_paper_trades,
    load_paper_trades,
    paper_summary_zh,
    record_signal_candidates,
    sync_paper_trades,
)
from crypto_signal_bot.strategies import evaluate_observation_candidate, evaluate_pullback_long

logger = logging.getLogger(__name__)

EXCLUDED_DYNAMIC_ALT_SYMBOLS = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "BUSDUSDT",
    "USDPUSDT",
    "BTCDOMUSDT",
    "XAUUSDT",
    "XAGUSDT",
    "CLUSDT",
    "BZUSDT",
}


DETAIL_LABELS = {
    "trend": "趋势",
    "no_uptrend": "未形成上涨趋势",
    "pullback": "回调",
    "no_pullback": "未出现有效回调",
    "reclaim": "重新站上均线",
    "momo": "短线动能",
    "vol": "成交量",
    "insufficient_history": "历史K线不足",
    "indicators_not_ready": "指标尚未就绪",
}


@dataclass(frozen=True)
class MarketTestRow:
    symbol: str
    interval: str
    candles: int
    last_close: float
    score: int
    detail: str

    def format_line(self) -> str:
        return (
            f"{self.symbol} {self.interval}：最新收盘价={self.last_close:g}，"
            f"策略评分={self.score}，信号详情={_format_detail_zh(self.detail)}"
        )


@dataclass(frozen=True)
class DynamicAltCandidate:
    symbol: str
    entry_day: str
    volume_ratio: float
    quote_volume: float
    daily_move_pct: float
    daily_range_pct: float
    rank_score: float

    def format_line(self) -> str:
        return (
            f"{self.symbol}({self.entry_day})：昨日成交额约 {self.quote_volume / 1_000_000:.1f}M USDT，"
            f"放大 {self.volume_ratio:.2f} 倍，涨跌幅 {self.daily_move_pct * 100:.2f}%，"
            f"日内振幅 {self.daily_range_pct * 100:.2f}%"
        )


def _format_detail_zh(detail: str) -> str:
    parts: list[str] = []
    for raw in detail.split(";"):
        if not raw:
            continue
        if raw in DETAIL_LABELS:
            parts.append(DETAIL_LABELS[raw])
            continue
        prefix = raw.rstrip("0123456789")
        value = raw[len(prefix) :]
        label = DETAIL_LABELS.get(prefix, raw)
        parts.append(f"{label}+{value}" if value else label)
    return "，".join(parts) if parts else "无"


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = symbol.strip().upper()
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _parse_symbol_thresholds(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in _parse_csv(raw):
        if ":" not in item:
            continue
        symbol, value = item.split(":", 1)
        try:
            threshold = int(value.strip())
        except ValueError:
            continue
        out[symbol.strip().upper()] = max(0, min(100, threshold))
    return out


def _normalize_direction(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"long", "buy", "l", "多", "做多", "做多观察"}:
        return "做多观察"
    if value in {"short", "sell", "s", "空", "做空", "做空观察"}:
        return "做空观察"
    return None


def _parse_direction_thresholds(raw: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for item in _parse_csv(raw):
        parts = item.split(":")
        if len(parts) != 3:
            continue
        symbol, direction_raw, value = parts
        direction = _normalize_direction(direction_raw)
        if direction is None:
            continue
        try:
            threshold = int(value.strip())
        except ValueError:
            continue
        out[(symbol.strip().upper(), direction)] = max(0, min(100, threshold))
    return out


def _threshold_for_symbol(symbol: str, default: int, overrides: dict[str, int]) -> int:
    return overrides.get(symbol.strip().upper(), default)


def _threshold_for_candidate(
    symbol: str,
    direction: str,
    default: int,
    symbol_overrides: dict[str, int],
    direction_overrides: dict[tuple[str, str], int],
) -> int:
    symbol_key = symbol.strip().upper()
    base = _threshold_for_symbol(symbol_key, default, symbol_overrides)
    return direction_overrides.get((symbol_key, direction), base)


def _opposite_direction(a: str, b: str) -> bool:
    return {a, b} == {"做多观察", "做空观察"}


def _find_matching_active_trade(trades: Sequence[PaperTrade], candidate) -> PaperTrade | None:
    symbol = candidate.symbol.strip().upper()
    direction = candidate.direction
    best: PaperTrade | None = None
    best_time = -1
    for trade in trades:
        if trade.status not in {PENDING, OPEN}:
            continue
        if trade.symbol != symbol or trade.direction != direction:
            continue
        trigger_gap = abs(trade.trigger_price - candidate.trigger_price) / max(candidate.trigger_price, 1e-9)
        stop_gap = abs(trade.stop_loss - candidate.stop_loss) / max(candidate.stop_loss, 1e-9)
        if trigger_gap > 0.001 or stop_gap > 0.001:
            continue
        if trade.signal_time_ms > best_time:
            best = trade
            best_time = trade.signal_time_ms
    return best


def _format_ms_local(time_ms: int | None) -> str:
    if not time_ms:
        return "-"
    return datetime.fromtimestamp(time_ms / 1000, UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _enhanced_confirmation_message(candidate, trade: PaperTrade) -> str:
    status = "待触发" if trade.status == PENDING else "持仓中"
    entry_hint = ""
    if trade.status == OPEN and trade.entry_time_ms is not None:
        entry_hint = f"\n入场价：{trade.entry_price:g}（{_format_ms_local(trade.entry_time_ms)}）"
    pnl_hint = ""
    if trade.status == OPEN and trade.entry_price > 0:
        pnl_hint = f"\n浮动盈亏：{trade.unrealized_pnl_pct * 100:.2f}%（最新价 {trade.last_price:g}）"

    return (
        f"增强确认：{candidate.symbol} {candidate.direction}\n\n"
        f"状态：{status}｜评分仍为 {candidate.score}/100\n"
        f"当前价：{candidate.current_price:g}\n"
        f"触发价：{candidate.trigger_price:g}\n"
        f"止损：{candidate.stop_loss:g}\n"
        f"TP1：{candidate.take_profit_1:g}｜TP2：{candidate.take_profit_2:g}\n"
        f"建议持仓：约 {candidate.expected_hold_hours:g} 小时｜目标盈亏比 {candidate.target_rr:g}:1"
        f"{entry_hint}{pnl_hint}\n\n"
        "说明：该币种同方向信号与上次一致，本轮不重复发送开仓通知。"
    )


def _paper_event_messages(before: Sequence[PaperTrade], after: Sequence[PaperTrade]) -> list[str]:
    before_by_id = {t.paper_id: t for t in before}
    out: list[str] = []
    for trade in after:
        prev = before_by_id.get(trade.paper_id)
        if prev is None:
            continue
        if not prev.tp1_hit and trade.tp1_hit:
            out.append(
                (
                    f"TP1触发：{trade.symbol} {trade.direction}\n\n"
                    f"入场：{trade.entry_price:g}｜TP1：{trade.tp1_price:g}\n"
                    f"最新价：{trade.last_price:g}｜浮动：{trade.unrealized_pnl_pct * 100:.2f}%\n"
                    f"当前止损(已抬高)：{trade.active_stop_price:g}\n"
                    f"触发时间：{_format_ms_local(trade.tp1_time_ms)}\n\n"
                    "建议：可以把止损抬到保本/小盈利区间，后续同币种同方向不再重复推送开仓提醒。"
                )
            )
        if prev.status != CLOSED and trade.status == CLOSED:
            extra = ""
            if trade.outcome in {"到期平仓", "时间止损"} and not trade.tp1_hit:
                extra = "\n\n建议：持仓超时且动能不足/横盘，建议离场并等待下一次高质量信号。"
            out.append(
                (
                    f"平仓提醒：{trade.symbol} {trade.direction}\n\n"
                    f"原因：{trade.outcome or '-'}\n"
                    f"入场：{trade.entry_price:g}（{_format_ms_local(trade.entry_time_ms)}）\n"
                    f"出场：{trade.exit_price:g}（{_format_ms_local(trade.exit_time_ms)}）\n"
                    f"收益：{trade.pnl_pct * 100:.2f}%（含手续费/资金费率）"
                    f"{extra}"
                )
            )
    return out


def _recent_return(candles: list[Candle], lookback: int = 6) -> float | None:
    if len(candles) <= lookback:
        return None
    old = candles[-lookback - 1].close
    if old <= 0:
        return None
    return (candles[-1].close - old) / old


def _day_after_open_time(open_time_ms: int) -> str:
    return (datetime.fromtimestamp(open_time_ms / 1000, UTC).date() + timedelta(days=1)).isoformat()


def _day_from_ms(time_ms: int) -> str:
    return datetime.fromtimestamp(time_ms / 1000, UTC).date().isoformat()


def _quote_volume(candle: Candle) -> float:
    return candle.volume * candle.close


def _format_big_number(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    v = float(value)
    abs_v = abs(v)
    if abs_v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.{digits}f}T"
    if abs_v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.{digits}f}B"
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.{digits}f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.{digits}f}K"
    return f"{v:.{digits}f}"


def _closed_candles(candles: list[Candle]) -> list[Candle]:
    """Drop the current still-forming bar when REST returns it (close_time in the future)."""
    if len(candles) < 2:
        return list(candles)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    if candles[-1].close_time > now_ms:
        return list(candles[:-1])
    return list(candles)


def _market_state_zh(trend_candles: list[Candle]) -> str:
    """Rough structure classification for hourly market summary: 上升/下跌/横盘."""
    candles = _closed_candles(trend_candles)
    closes_ = closes(candles)
    e20 = ema(closes_, 20)
    e50 = ema(closes_, 50)
    s20 = ema_slope(closes_, 20, 3)
    s50 = ema_slope(closes_, 50, 3)
    if None in {e20, e50, s20, s50}:
        return "未知"
    price = candles[-1].close
    # 与策略里的趋势过滤保持一致：允许极轻微回撤，只要结构仍然偏多/偏空。
    eps20 = (e20 or 0.0) * 0.00012
    eps50 = (e50 or 0.0) * 0.00008
    if price > (e20 or 0.0) and (e20 or 0.0) >= (e50 or 0.0) and (s20 or 0.0) > -eps20 and (s50 or 0.0) > -eps50:
        return "上升"
    if price < (e20 or 0.0) and (e20 or 0.0) <= (e50 or 0.0) and (s20 or 0.0) < eps20 and (s50 or 0.0) < eps50:
        return "下跌"
    return "横盘"


def _market_summary_message(symbols: list[str], trend_by_symbol: dict[str, list[Candle]]) -> str:
    now_local = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "每小时市场总结",
        f"时间：{now_local}",
        f"本轮观察池：{len(symbols)}",
        "",
    ]
    for symbol in symbols:
        candles = trend_by_symbol.get(symbol)
        if not candles:
            lines.append(f"{symbol}：数据缺失")
            continue
        closed = _closed_candles(candles)
        state = _market_state_zh(closed)
        last_price = closed[-1].close if closed else candles[-1].close
        r6 = _recent_return(closed, 6) if closed else None
        r24 = _recent_return(closed, 24) if closed else None
        vol24 = sum(_quote_volume(c) for c in (closed[-24:] if len(closed) >= 24 else closed))
        # 1小时量能相对过去20小时均值倍数
        vol1 = _quote_volume(closed[-1]) if closed else _quote_volume(candles[-1])
        base = None
        if len(closed) >= 20:
            avg20 = sum(_quote_volume(c) for c in closed[-20:]) / 20.0
            if avg20 > 0:
                base = vol1 / avg20
        r6s = "-" if r6 is None else f"{r6 * 100:+.2f}%"
        r24s = "-" if r24 is None else f"{r24 * 100:+.2f}%"
        vol24s = _format_big_number(vol24, digits=2)
        mults = "-" if base is None else f"{base:.2f}x"
        lines.append(f"{symbol}：{state}｜现价 {last_price:g}｜6h {r6s}｜24h {r24s}｜24h成交额 {vol24s}｜1h量能 {mults}")
    return "\n".join(lines)


def _daily_move_pct(candle: Candle) -> float:
    if candle.open <= 0:
        return 0.0
    return (candle.close - candle.open) / candle.open


def _daily_range_pct(candle: Candle) -> float:
    if candle.close <= 0:
        return 0.0
    return (candle.high - candle.low) / candle.close


def _dynamic_alt_candidate_from_previous_day(
    *,
    symbol: str,
    previous_day: Candle,
    baseline_days: list[Candle],
    entry_day: str,
    min_volume_ratio: float,
    min_quote_volume: float,
    min_daily_move_pct: float,
    max_daily_move_pct: float,
    min_daily_range_pct: float,
    max_daily_range_pct: float,
) -> DynamicAltCandidate | None:
    if not baseline_days:
        return None
    quote_volume = _quote_volume(previous_day)
    baseline_quote_volume = sum(_quote_volume(c) for c in baseline_days) / len(baseline_days)
    if baseline_quote_volume <= 0:
        return None
    volume_ratio = quote_volume / baseline_quote_volume
    daily_move = _daily_move_pct(previous_day)
    daily_range = _daily_range_pct(previous_day)

    if quote_volume < min_quote_volume:
        return None
    if volume_ratio < min_volume_ratio:
        return None
    if daily_move < min_daily_move_pct:
        return None
    if abs(daily_move) > max_daily_move_pct:
        return None
    if not min_daily_range_pct <= daily_range <= max_daily_range_pct:
        return None

    rank_score = volume_ratio * (quote_volume / 1_000_000_000) * max(0.5, daily_range / 0.05)
    return DynamicAltCandidate(
        symbol=symbol.strip().upper(),
        entry_day=entry_day,
        volume_ratio=volume_ratio,
        quote_volume=quote_volume,
        daily_move_pct=daily_move,
        daily_range_pct=daily_range,
        rank_score=rank_score,
    )


def _latest_dynamic_alt_candidate(
    *,
    symbol: str,
    daily_candles: list[Candle],
    as_of_ms: int,
    lookback_days: int,
    min_volume_ratio: float,
    min_quote_volume: float,
    min_daily_move_pct: float,
    max_daily_move_pct: float,
    min_daily_range_pct: float,
    max_daily_range_pct: float,
) -> DynamicAltCandidate | None:
    closed = [c for c in sorted(daily_candles, key=lambda c: c.open_time) if c.close_time <= as_of_ms]
    if len(closed) < lookback_days + 1:
        return None
    previous_day = closed[-1]
    baseline = closed[-lookback_days - 1 : -1]
    return _dynamic_alt_candidate_from_previous_day(
        symbol=symbol,
        previous_day=previous_day,
        baseline_days=baseline,
        entry_day=_day_after_open_time(previous_day.open_time),
        min_volume_ratio=min_volume_ratio,
        min_quote_volume=min_quote_volume,
        min_daily_move_pct=min_daily_move_pct,
        max_daily_move_pct=max_daily_move_pct,
        min_daily_range_pct=min_daily_range_pct,
        max_daily_range_pct=max_daily_range_pct,
    )


def _dynamic_alt_candidates_for_days(
    *,
    symbol: str,
    daily_candles: list[Candle],
    start_day: str,
    end_day: str,
    lookback_days: int,
    min_volume_ratio: float,
    min_quote_volume: float,
    min_daily_move_pct: float,
    max_daily_move_pct: float,
    min_daily_range_pct: float,
    max_daily_range_pct: float,
) -> list[DynamicAltCandidate]:
    candles = sorted(daily_candles, key=lambda c: c.open_time)
    out: list[DynamicAltCandidate] = []
    for idx in range(lookback_days, len(candles)):
        previous_day = candles[idx]
        entry_day = _day_after_open_time(previous_day.open_time)
        if entry_day < start_day or entry_day > end_day:
            continue
        candidate = _dynamic_alt_candidate_from_previous_day(
            symbol=symbol,
            previous_day=previous_day,
            baseline_days=candles[idx - lookback_days : idx],
            entry_day=entry_day,
            min_volume_ratio=min_volume_ratio,
            min_quote_volume=min_quote_volume,
            min_daily_move_pct=min_daily_move_pct,
            max_daily_move_pct=max_daily_move_pct,
            min_daily_range_pct=min_daily_range_pct,
            max_daily_range_pct=max_daily_range_pct,
        )
        if candidate is not None:
            out.append(candidate)
    return out


async def _select_dynamic_alt_candidates(
    *,
    fixed_symbols: list[str],
    top_limit: int,
    limit: int,
    lookback_days: int,
    min_volume_ratio: float,
    min_quote_volume: float,
    min_daily_move_pct: float,
    max_daily_move_pct: float,
    min_daily_range_pct: float,
    max_daily_range_pct: float,
) -> list[DynamicAltCandidate]:
    if limit <= 0 or top_limit <= 0:
        return []
    excluded = {s.strip().upper() for s in fixed_symbols} | EXCLUDED_DYNAMIC_ALT_SYMBOLS
    universe = [
        symbol
        for symbol in await fetch_top_usdm_symbols_by_quote_volume(limit=top_limit)
        if symbol not in excluded
    ]
    as_of_ms = int(datetime.now(UTC).timestamp() * 1000)
    semaphore = asyncio.Semaphore(6)

    async def load(symbol: str) -> DynamicAltCandidate | None:
        try:
            async with semaphore:
                klines = await fetch_usdm_klines(symbol=symbol, interval="1d", limit=lookback_days + 3)
        except Exception as exc:
            logger.info("%s 动态山寨日线筛选失败：%s", symbol, exc)
            return None
        daily = [Candle.from_kline(k) for k in klines]
        return _latest_dynamic_alt_candidate(
            symbol=symbol,
            daily_candles=daily,
            as_of_ms=as_of_ms,
            lookback_days=lookback_days,
            min_volume_ratio=min_volume_ratio,
            min_quote_volume=min_quote_volume,
            min_daily_move_pct=min_daily_move_pct,
            max_daily_move_pct=max_daily_move_pct,
            min_daily_range_pct=min_daily_range_pct,
            max_daily_range_pct=max_daily_range_pct,
        )

    results = await asyncio.gather(*(load(symbol) for symbol in universe))
    candidates = [candidate for candidate in results if candidate is not None]
    return sorted(candidates, key=lambda c: (c.rank_score, c.quote_volume), reverse=True)[:limit]


async def _select_dynamic_alt_entry_days(
    *,
    fixed_symbols: list[str],
    days: int,
    top_limit: int,
    limit: int,
    lookback_days: int,
    min_volume_ratio: float,
    min_quote_volume: float,
    min_daily_move_pct: float,
    max_daily_move_pct: float,
    min_daily_range_pct: float,
    max_daily_range_pct: float,
) -> dict[str, set[str]]:
    if limit <= 0 or top_limit <= 0:
        return {}
    excluded = {s.strip().upper() for s in fixed_symbols} | EXCLUDED_DYNAMIC_ALT_SYMBOLS
    universe = [
        symbol
        for symbol in await fetch_top_usdm_symbols_by_quote_volume(limit=top_limit)
        if symbol not in excluded
    ]
    end = datetime.now(UTC)
    start = end - timedelta(days=days + lookback_days + 3)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    start_day = (end - timedelta(days=days)).date().isoformat()
    end_day = end.date().isoformat()
    by_day: dict[str, list[DynamicAltCandidate]] = defaultdict(list)

    for symbol in universe:
        try:
            klines = await fetch_usdm_klines_range(
                symbol=symbol,
                interval="1d",
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
        except Exception as exc:
            logger.info("%s 动态山寨回测日线筛选失败：%s", symbol, exc)
            continue
        daily = [Candle.from_kline(k) for k in klines]
        for candidate in _dynamic_alt_candidates_for_days(
            symbol=symbol,
            daily_candles=daily,
            start_day=start_day,
            end_day=end_day,
            lookback_days=lookback_days,
            min_volume_ratio=min_volume_ratio,
            min_quote_volume=min_quote_volume,
            min_daily_move_pct=min_daily_move_pct,
            max_daily_move_pct=max_daily_move_pct,
            min_daily_range_pct=min_daily_range_pct,
            max_daily_range_pct=max_daily_range_pct,
        ):
            by_day[candidate.entry_day].append(candidate)

    allowed: dict[str, set[str]] = defaultdict(set)
    for entry_day, candidates in by_day.items():
        ranked = sorted(candidates, key=lambda c: (c.rank_score, c.quote_volume), reverse=True)
        for candidate in ranked[:limit]:
            allowed[candidate.symbol].add(entry_day)
    return dict(allowed)


async def _optional_ping(notifier: TelegramNotifier) -> None:
    await notifier.send_text("加密货币信号机器人：Telegram 通知测试成功。")


async def _market_test(
    *,
    symbols: list[str],
    intervals: list[str],
    limit: int,
    notifier: TelegramNotifier | None,
) -> list[MarketTestRow]:
    rows: list[MarketTestRow] = []
    for symbol in symbols:
        for interval in intervals:
            klines = await fetch_usdm_klines(symbol=symbol, interval=interval, limit=limit)
            candles = [Candle.from_kline(k) for k in klines]
            result = evaluate_pullback_long(candles)
            row = MarketTestRow(
                symbol=symbol.strip().upper(),
                interval=interval.strip(),
                candles=len(candles),
                last_close=candles[-1].close,
                score=result.score,
                detail=result.detail,
            )
            rows.append(row)
            logger.info(row.format_line())

    if notifier is not None:
        body = "加密货币信号机器人市场检测\n" + "\n".join(row.format_line() for row in rows)
        await notifier.send_text(body)

    return rows


async def _scan_market(
    *,
    fixed_symbols: list[str],
    top_volume_limit: int,
    score_threshold: int,
    expected_hold_hours: float,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    target_rr: float,
    expire_minutes: int,
    symbol_thresholds: dict[str, int],
    direction_thresholds: dict[tuple[str, str], int],
    dynamic_alt_limit: int,
    dynamic_alt_top_limit: int,
    dynamic_alt_lookback_days: int,
    dynamic_alt_volume_ratio: float,
    dynamic_alt_min_quote_volume: float,
    dynamic_alt_min_daily_move_pct: float,
    dynamic_alt_max_daily_move_pct: float,
    dynamic_alt_min_daily_range_pct: float,
    dynamic_alt_max_daily_range_pct: float,
    dynamic_alt_threshold: int,
    dynamic_alt_long_only: bool,
    conflict_threshold: int,
    relative_rank_limit: int,
    weekly_filter: bool,
    paper_record: bool,
    paper_path: str,
    confirm_minutes: int,
    paper_entry_mode: str,
    symbol_cooldown_minutes: int,
    notifier: TelegramNotifier | None,
    send_market_summary: bool = False,
) -> None:
    top_symbols = await fetch_top_usdm_symbols_by_quote_volume(limit=top_volume_limit)
    dynamic_alt_candidates = await _select_dynamic_alt_candidates(
        fixed_symbols=fixed_symbols + top_symbols,
        top_limit=dynamic_alt_top_limit,
        limit=dynamic_alt_limit,
        lookback_days=dynamic_alt_lookback_days,
        min_volume_ratio=dynamic_alt_volume_ratio,
        min_quote_volume=dynamic_alt_min_quote_volume,
        min_daily_move_pct=dynamic_alt_min_daily_move_pct,
        max_daily_move_pct=dynamic_alt_max_daily_move_pct,
        min_daily_range_pct=dynamic_alt_min_daily_range_pct,
        max_daily_range_pct=dynamic_alt_max_daily_range_pct,
    )
    dynamic_alt_symbols = [candidate.symbol for candidate in dynamic_alt_candidates]
    dynamic_alt_symbol_set = set(dynamic_alt_symbols)
    if dynamic_alt_candidates:
        logger.info(
            "动态山寨机会池：%s",
            "；".join(candidate.format_line() for candidate in dynamic_alt_candidates),
        )
    symbols = _dedupe_symbols(fixed_symbols + top_symbols + dynamic_alt_symbols)
    logger.info("本轮观察池：%s", "，".join(symbols))
    market_candles = None
    market_entry_candles = None
    if any(symbol != "BTCUSDT" for symbol in symbols):
        btc_market_klines = await fetch_usdm_klines(symbol="BTCUSDT", interval="1h", limit=360)
        market_candles = [Candle.from_kline(k) for k in btc_market_klines]
        btc_entry_klines = await fetch_usdm_klines(symbol="BTCUSDT", interval="15m", limit=180)
        market_entry_candles = [Candle.from_kline(k) for k in btc_entry_klines]

    candidates = {}
    trend_by_symbol: dict[str, list[Candle]] = {}
    for symbol in symbols:
        try:
            trend_klines = await fetch_usdm_klines(symbol=symbol, interval="1h", limit=360)
            entry_klines = await fetch_usdm_klines(symbol=symbol, interval="15m", limit=180)
        except Exception as exc:
            logger.warning("%s 行情拉取失败：%s", symbol, exc)
            continue
        trend_candles = [Candle.from_kline(k) for k in trend_klines]
        entry_candles = [Candle.from_kline(k) for k in entry_klines]
        trend_by_symbol[symbol] = trend_candles

        candidate = evaluate_observation_candidate(
            symbol=symbol,
            trend_candles=trend_candles,
            entry_candles=entry_candles,
            expected_hold_hours=expected_hold_hours,
            min_hold_hours=min_hold_hours,
            max_hold_hours=max_hold_hours,
            dynamic_hold=dynamic_hold,
            target_rr=target_rr,
            expires_after_minutes=expire_minutes,
            market_candles=None if symbol == "BTCUSDT" else market_candles,
            market_entry_candles=None if symbol == "BTCUSDT" else market_entry_candles,
            weekly_filter=weekly_filter,
        )
        logger.info(candidate.summary_line())
        candidates[symbol] = candidate

    summary_msg: str | None = None
    if send_market_summary and notifier is not None:
        try:
            summary_msg = _market_summary_message(symbols, trend_by_symbol)
        except Exception as exc:  # pragma: no cover
            logger.warning("市场总结生成失败：%s", exc)
            summary_msg = None

    qualified = []
    btc_candidate = candidates.get("BTCUSDT")
    long_ranked = sorted(
        (
            (symbol, ret)
            for symbol, candles in trend_by_symbol.items()
            if (ret := _recent_return(candles, 6)) is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    short_ranked = sorted(
        (
            (symbol, -ret)
            for symbol, candles in trend_by_symbol.items()
            if (ret := _recent_return(candles, 6)) is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    long_top = {symbol for symbol, _ in long_ranked[:relative_rank_limit]} if relative_rank_limit > 0 else set()
    short_top = {symbol for symbol, _ in short_ranked[:relative_rank_limit]} if relative_rank_limit > 0 else set()
    for symbol, candidate in candidates.items():
        threshold = _threshold_for_candidate(
            symbol,
            candidate.direction,
            score_threshold,
            symbol_thresholds,
            direction_thresholds,
        )
        if symbol in dynamic_alt_symbol_set:
            threshold = max(threshold, dynamic_alt_threshold)
            if dynamic_alt_long_only and candidate.direction != "做多观察":
                logger.info("%s 属于动态山寨机会池，但当前不是做多候选，本轮只记录", symbol)
                continue
        if (
            symbol != "BTCUSDT"
            and btc_candidate is not None
            and btc_candidate.score >= conflict_threshold
            and candidate.direction in {"做多观察", "做空观察"}
            and btc_candidate.direction in {"做多观察", "做空观察"}
            and _opposite_direction(candidate.direction, btc_candidate.direction)
        ):
            logger.info("%s 与 BTC 高分相反候选冲突，本轮不推送", symbol)
            continue
        if relative_rank_limit > 0 and candidate.direction == "做多观察" and symbol not in long_top:
            logger.info("%s 未进入近6小时相对强势前%s，本轮做多只记录", symbol, relative_rank_limit)
            continue
        if relative_rank_limit > 0 and candidate.direction == "做空观察" and symbol not in short_top:
            logger.info("%s 未进入近6小时相对弱势前%s，本轮做空只记录", symbol, relative_rank_limit)
            continue
        if candidate.score >= threshold:
            qualified.append(candidate)

    recorded: list[PaperTrade] = []
    if paper_record and qualified:
        recorded = record_signal_candidates(
            qualified,
            paper_path,
            confirm_minutes=confirm_minutes,
            entry_mode=paper_entry_mode,
            cooldown_minutes=symbol_cooldown_minutes,
        )
        logger.info(
            "虚拟盘记录：新增 %s 条，重复/冷却跳过 %s 条，账本=%s",
            len(recorded),
            len(qualified) - len(recorded),
            paper_path,
        )

    if notifier is None:
        return

    if summary_msg is not None:
        await notifier.send_text(summary_msg)

    if not qualified:
        # 不达标不通知：常驻运行时避免刷屏，只在有“有效动作”时推送。
        logger.info("本轮无达标候选（score_threshold=%s），不发送通知。", score_threshold)
        return

    trades = load_paper_trades(paper_path) if paper_record else []
    recorded_keys = {(t.symbol, t.direction) for t in recorded}
    entry_messages: list[str] = []
    confirm_messages: list[str] = []
    suppressed = 0
    for candidate in qualified:
        key = (candidate.symbol.strip().upper(), candidate.direction)
        if key in recorded_keys:
            entry_messages.append(candidate.to_telegram_message())
            continue
        if not paper_record:
            entry_messages.append(candidate.to_telegram_message())
            continue
        matched = _find_matching_active_trade(trades, candidate)
        if matched is not None and not matched.tp1_hit:
            confirm_messages.append(_enhanced_confirmation_message(candidate, matched))
        else:
            suppressed += 1

    await notifier.send_text(
        f"观察型市场扫描完成\n"
        f"观察池数量：{len(symbols)}\n"
        f"推送阈值：{score_threshold}\n"
        f"达标备案：{len(qualified)}\n"
        f"新增开仓通知：{len(entry_messages)}\n"
        f"增强确认：{len(confirm_messages)}"
    )
    for msg in entry_messages:
        await notifier.send_text(msg)
    for msg in confirm_messages:
        await notifier.send_text(msg)

    if suppressed:
        logger.info("通知去重：抑制重复=%s", suppressed)


async def _run_live_loop(
    *,
    fixed_symbols: list[str],
    top_volume_limit: int,
    score_threshold: int,
    expected_hold_hours: float,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    target_rr: float,
    expire_minutes: int,
    symbol_thresholds: dict[str, int],
    direction_thresholds: dict[tuple[str, str], int],
    dynamic_alt_limit: int,
    dynamic_alt_top_limit: int,
    dynamic_alt_lookback_days: int,
    dynamic_alt_volume_ratio: float,
    dynamic_alt_min_quote_volume: float,
    dynamic_alt_min_daily_move_pct: float,
    dynamic_alt_max_daily_move_pct: float,
    dynamic_alt_min_daily_range_pct: float,
    dynamic_alt_max_daily_range_pct: float,
    dynamic_alt_threshold: int,
    dynamic_alt_long_only: bool,
    conflict_threshold: int,
    relative_rank_limit: int,
    weekly_filter: bool,
    paper_path: str,
    paper_entry_mode: str,
    confirm_minutes: int,
    paper_export_path: str,
    paper_export_every_hours: int,
    paper_snapshot_dir: str,
    loop_minutes: int,
    market_summary_every_scans: int,
    symbol_cooldown_minutes: int,
    fee_rate: float,
    funding_rate_8h: float,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    notifier: TelegramNotifier | None,
) -> None:
    logger.info("常驻运行已启动：间隔=%s分钟；虚拟盘=%s；导出=%s", loop_minutes, paper_path, paper_export_path)
    snapshot_dir = Path(paper_snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    export_every = max(0, int(paper_export_every_hours))
    next_export_at = datetime.now(UTC) + timedelta(hours=export_every) if export_every else None
    loop_no = 0
    consecutive_network_errors = 0
    while True:
        loop_no += 1
        logger.info("常驻运行第 %s 轮开始（UTC=%s）", loop_no, datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"))
        try:
            send_summary = (
                notifier is not None
                and market_summary_every_scans > 0
                and loop_no % market_summary_every_scans == 0
            )
            await _scan_market(
                fixed_symbols=fixed_symbols,
                top_volume_limit=top_volume_limit,
                score_threshold=score_threshold,
                expected_hold_hours=expected_hold_hours,
                min_hold_hours=min_hold_hours,
                max_hold_hours=max_hold_hours,
                dynamic_hold=dynamic_hold,
                target_rr=target_rr,
                expire_minutes=expire_minutes,
                symbol_thresholds=symbol_thresholds,
                direction_thresholds=direction_thresholds,
                dynamic_alt_limit=dynamic_alt_limit,
                dynamic_alt_top_limit=dynamic_alt_top_limit,
                dynamic_alt_lookback_days=dynamic_alt_lookback_days,
                dynamic_alt_volume_ratio=dynamic_alt_volume_ratio,
                dynamic_alt_min_quote_volume=dynamic_alt_min_quote_volume,
                dynamic_alt_min_daily_move_pct=dynamic_alt_min_daily_move_pct,
                dynamic_alt_max_daily_move_pct=dynamic_alt_max_daily_move_pct,
                dynamic_alt_min_daily_range_pct=dynamic_alt_min_daily_range_pct,
                dynamic_alt_max_daily_range_pct=dynamic_alt_max_daily_range_pct,
                dynamic_alt_threshold=dynamic_alt_threshold,
                dynamic_alt_long_only=dynamic_alt_long_only,
                conflict_threshold=conflict_threshold,
                relative_rank_limit=relative_rank_limit,
                weekly_filter=weekly_filter,
                paper_record=True,
                paper_path=paper_path,
                confirm_minutes=confirm_minutes,
                paper_entry_mode=paper_entry_mode,
                symbol_cooldown_minutes=symbol_cooldown_minutes,
                notifier=notifier,
                send_market_summary=send_summary,
            )
        except Exception as exc:
            logger.exception("常驻运行扫描失败：%s", exc)
            snap = _export_snapshot_on_issue(snapshot_dir, paper_export_path, paper_path, reason="scan_error")
            if _is_network_error(exc):
                consecutive_network_errors += 1
                if consecutive_network_errors >= 2:
                    await _notify_and_stop_on_network_error(notifier, exc, snap)
                await _maybe_notify_network_error(notifier, exc, snap, consecutive_network_errors, loop_minutes)
                logger.info("网络异常，本轮扫描暂停，等待 %s 分钟后重试。", loop_minutes)
                await asyncio.sleep(loop_minutes * 60)
                continue

        try:
            before_trades = load_paper_trades(paper_path)
            result = await sync_paper_trades(
                paper_path,
                fee_rate=fee_rate,
                funding_rate_8h=funding_rate_8h,
                time_stop_minutes=time_stop_minutes,
                min_progress_r=min_progress_r,
                r_trailing_enabled=r_trailing_enabled,
                trailing_trigger_pct=trailing_trigger_pct,
                trailing_lock_pct=trailing_lock_pct,
            )
            consecutive_network_errors = 0
            if notifier is not None and result.changed_count:
                for msg in _paper_event_messages(before_trades, result.trades):
                    await notifier.send_text(msg)
            now_utc = datetime.now(UTC)
            exported: Path | None = None
            if export_every == 0:
                exported = export_paper_trades(paper_export_path, result.trades)
            elif next_export_at is not None and now_utc >= next_export_at:
                exported = export_paper_trades(_dated_export_path(paper_export_path, now_utc), result.trades)
                next_export_at = now_utc + timedelta(hours=export_every)
            if exported is not None:
                logger.info("虚拟盘已同步：更新 %s 条；已导出：%s", result.changed_count, exported)
            else:
                logger.info("虚拟盘已同步：更新 %s 条；本轮未到导出时间", result.changed_count)
        except Exception as exc:
            logger.exception("常驻运行虚拟盘同步/导出失败：%s", exc)
            snap = _export_snapshot_on_issue(snapshot_dir, paper_export_path, paper_path, reason="sync_error")
            if _is_network_error(exc):
                consecutive_network_errors += 1
                if consecutive_network_errors >= 2:
                    await _notify_and_stop_on_network_error(notifier, exc, snap)
                await _maybe_notify_network_error(notifier, exc, snap, consecutive_network_errors, loop_minutes)
                logger.info("网络异常，本轮同步暂停，等待 %s 分钟后重试。", loop_minutes)
                await asyncio.sleep(loop_minutes * 60)
                continue

        next_run = datetime.now(UTC) + timedelta(minutes=loop_minutes)
        logger.info("本轮结束，等待 %s 分钟后继续（预计UTC=%s）", loop_minutes, next_run.strftime("%Y-%m-%d %H:%M:%S"))
        await asyncio.sleep(loop_minutes * 60)


def _dated_export_path(base: str, now_utc: datetime) -> str:
    p = Path(base)
    stamp = now_utc.strftime("%Y-%m-%d")
    return str(p.with_name(f"{p.stem}_{stamp}{p.suffix}"))


def _timestamped_export_path(base: str, now_utc: datetime, reason: str) -> str:
    p = Path(base)
    stamp = now_utc.strftime("%Y-%m-%d_%H%M%S")
    safe_reason = "".join(ch for ch in (reason or "snapshot") if ch.isalnum() or ch in {"_", "-"}).strip("_-")
    suffix = f"_{safe_reason}" if safe_reason else ""
    return str(p.with_name(f"{p.stem}_{stamp}{suffix}{p.suffix}"))


def _export_snapshot_on_issue(snapshot_dir: Path, base_export_path: str, ledger_path: str, *, reason: str) -> Path | None:
    try:
        trades = load_paper_trades(ledger_path)
        now_utc = datetime.now(UTC)
        base = Path(base_export_path)
        target = snapshot_dir / Path(_timestamped_export_path(base.name, now_utc, reason))
        out = export_paper_trades(target, trades)
        logger.info("检测到异常，已导出虚拟盘快照：%s", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.warning("异常快照导出失败：%s", exc)
        return None


def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, OSError):
        return True
    return False


@dataclass(frozen=True)
class DirectionAccuracy:
    total: int
    wins: int
    ties: int
    avg_move_pct: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0


async def _direction_accuracy_1h(
    *,
    trades_csv: str,
    horizon_minutes: int,
    min_move_pct: float,
) -> None:
    """
    Evaluate directional correctness after a fixed horizon.

    Definition:
    - Take each executed trade from exported backtest CSV.
    - Find the first 5m candle whose close_time >= entry_time + horizon.
    - LONG is "correct" if price_change_pct > +min_move_pct.
    - SHORT is "correct" if price_change_pct < -min_move_pct.
    - Otherwise counted as tie/incorrect depending on sign; we track ties explicitly.
    """
    path = Path(trades_csv)
    if not path.exists():
        raise FileNotFoundError(f"找不到交易明细CSV：{trades_csv}")
    if horizon_minutes <= 0:
        raise ValueError("--direction-horizon-minutes 必须大于 0")
    if min_move_pct < 0:
        raise ValueError("--direction-min-move-pct 不能小于 0")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r]
    if not rows:
        raise ValueError(f"CSV 没有任何记录：{trades_csv}")

    # Parse trades and compute fetch range.
    horizon_ms = horizon_minutes * 60_000
    min_entry_ms = 2**63 - 1
    max_eval_ms = 0
    trades: list[tuple[str, str, int, float]] = []
    for r in rows:
        symbol = (r.get("交易对") or "").strip().upper()
        direction = (r.get("方向") or "").strip()
        entry_utc_text = (r.get("入场时间(UTC)") or "").strip()
        entry_price_text = (r.get("入场价") or "").strip()
        if not symbol or not direction or not entry_utc_text or not entry_price_text:
            continue
        dt = datetime.fromisoformat(entry_utc_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        entry_ms = int(dt.timestamp() * 1000)
        try:
            entry_price = float(entry_price_text)
        except ValueError:
            continue
        min_entry_ms = min(min_entry_ms, entry_ms)
        max_eval_ms = max(max_eval_ms, entry_ms + horizon_ms)
        trades.append((symbol, direction, entry_ms, entry_price))
    if not trades:
        raise ValueError("CSV 中没有可解析的交易记录。")

    # Load 5m klines once per symbol.
    by_symbol: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for symbol, direction, entry_ms, entry_price in trades:
        by_symbol[symbol].append((direction, entry_ms, entry_price))

    logger.info(
        "开始方向准确率评估：交易数=%s，周期=%s分钟，最小方向阈值=%.4f%%，交易对=%s",
        len(trades),
        horizon_minutes,
        min_move_pct,
        "，".join(sorted(by_symbol)),
    )

    candles_by_symbol: dict[str, tuple[list[int], list[float]]] = {}
    # Provide a small pad to avoid missing the last eval point.
    start_ms = max(0, min_entry_ms - 6 * 60_000)
    end_ms = max_eval_ms + 6 * 60_000
    for symbol in sorted(by_symbol):
        klines = await fetch_usdm_klines_range(
            symbol=symbol,
            interval="5m",
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        candles = [Candle.from_kline(k) for k in klines]
        close_times = [c.close_time for c in candles]
        closes_ = [c.close for c in candles]
        candles_by_symbol[symbol] = (close_times, closes_)
        logger.info("%s 5m K线载入完成：%s 根", symbol, len(candles))

    def evaluate(group: list[tuple[str, int, float]], series: tuple[list[int], list[float]]) -> DirectionAccuracy:
        close_times, closes_ = series
        total = 0
        wins = 0
        ties = 0
        sum_move = 0.0
        for direction, entry_ms, entry_price in group:
            eval_ms = entry_ms + horizon_ms
            idx = bisect_left(close_times, eval_ms)
            if idx >= len(closes_) or entry_price <= 0:
                continue
            price = closes_[idx]
            move = (price - entry_price) / entry_price
            total += 1
            # move in the "favorable" direction for averaging.
            favorable_move = move if direction == "做多观察" else -move
            sum_move += favorable_move * 100

            threshold = min_move_pct / 100.0
            if abs(move) <= threshold:
                ties += 1
                continue
            if direction == "做多观察":
                if move > threshold:
                    wins += 1
            elif direction == "做空观察":
                if move < -threshold:
                    wins += 1
        avg_move = (sum_move / total) if total else 0.0
        return DirectionAccuracy(total=total, wins=wins, ties=ties, avg_move_pct=avg_move)

    overall_total = 0
    overall_wins = 0
    overall_ties = 0
    overall_move_sum = 0.0

    print("")
    print(f"方向准确率评估（{horizon_minutes}分钟）")
    print(f"判定阈值：|涨跌幅| <= {min_move_pct:.4f}% 视为平局")
    print(f"数据来源：{trades_csv}")
    print("")
    for symbol in sorted(by_symbol):
        acc = evaluate(by_symbol[symbol], candles_by_symbol[symbol])
        overall_total += acc.total
        overall_wins += acc.wins
        overall_ties += acc.ties
        overall_move_sum += acc.avg_move_pct * acc.total
        print(
            f"{symbol}：胜率 {acc.win_rate * 100:.2f}%（{acc.wins}/{acc.total}），"
            f"平局 {acc.ties}，平均有利波动 {acc.avg_move_pct:+.3f}%"
        )

    overall_avg_move = (overall_move_sum / overall_total) if overall_total else 0.0
    overall_win_rate = (overall_wins / overall_total) if overall_total else 0.0
    print("")
    print(
        f"总体：胜率 {overall_win_rate * 100:.2f}%（{overall_wins}/{overall_total}），"
        f"平局 {overall_ties}，平均有利波动 {overall_avg_move:+.3f}%"
    )


async def _notify_and_stop_on_network_error(
    notifier: TelegramNotifier | None,
    exc: Exception,
    snapshot: Path | None,
) -> None:
    msg = f"网络异常，常驻监测已停止。\n错误：{type(exc).__name__}: {exc}"
    if snapshot is not None:
        msg += f"\n已导出快照：{snapshot}"
    logger.error(msg)
    if notifier is not None:
        try:
            await notifier.send_text(msg)
        except Exception:
            pass
    raise SystemExit(2)


async def _maybe_notify_network_error(
    notifier: TelegramNotifier | None,
    exc: Exception,
    snapshot: Path | None,
    consecutive_count: int,
    loop_minutes: int,
) -> None:
    # 仅在第一次网络异常时提示“将重试”，避免刷屏。
    if notifier is None or consecutive_count != 1:
        return
    msg = f"网络异常，本轮已暂停，将在下一轮重试（约 {loop_minutes} 分钟后）。\n错误：{type(exc).__name__}: {exc}"
    if snapshot is not None:
        msg += f"\n已导出快照：{snapshot}"
    msg += "\n提示：如果连续两轮仍出现网络异常，程序将自动停止。"
    try:
        await notifier.send_text(msg)
    except Exception:
        pass


async def _backtest(
    *,
    symbols: list[str],
    days: int,
    score_threshold: int,
    hold_hours: float,
    fee_rate: float,
    funding_rate_8h: float,
    target_rr: float,
    confirm_minutes: int,
    expire_minutes: int,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    time_stop_minutes: int,
    min_progress_r: float,
    r_trailing_enabled: bool,
    trailing_trigger_pct: float,
    trailing_lock_pct: float,
    weekly_filter: bool,
    symbol_thresholds: dict[str, int],
    direction_thresholds: dict[tuple[str, str], int],
    dynamic_alt_limit: int,
    dynamic_alt_top_limit: int,
    dynamic_alt_lookback_days: int,
    dynamic_alt_volume_ratio: float,
    dynamic_alt_min_quote_volume: float,
    dynamic_alt_min_daily_move_pct: float,
    dynamic_alt_max_daily_move_pct: float,
    dynamic_alt_min_daily_range_pct: float,
    dynamic_alt_max_daily_range_pct: float,
    dynamic_alt_threshold: int,
    dynamic_alt_long_only: bool,
    conflict_threshold: int,
    conflict_minutes: int,
    export_trades: str | None,
    notifier: TelegramNotifier | None,
) -> None:
    reports: list[str] = []
    summaries: list[BacktestSummary] = []
    fixed_symbols = _dedupe_symbols(symbols)
    dynamic_entry_days = await _select_dynamic_alt_entry_days(
        fixed_symbols=fixed_symbols,
        days=days,
        top_limit=dynamic_alt_top_limit,
        limit=dynamic_alt_limit,
        lookback_days=dynamic_alt_lookback_days,
        min_volume_ratio=dynamic_alt_volume_ratio,
        min_quote_volume=dynamic_alt_min_quote_volume,
        min_daily_move_pct=dynamic_alt_min_daily_move_pct,
        max_daily_move_pct=dynamic_alt_max_daily_move_pct,
        min_daily_range_pct=dynamic_alt_min_daily_range_pct,
        max_daily_range_pct=dynamic_alt_max_daily_range_pct,
    )
    dynamic_symbols = sorted(dynamic_entry_days)
    if dynamic_symbols:
        logger.info(
            "动态山寨回测池：%s",
            "，".join(f"{symbol}({len(days_set)}天)" for symbol, days_set in dynamic_entry_days.items()),
        )
    symbols = _dedupe_symbols(fixed_symbols + dynamic_symbols)
    for symbol in symbols:
        logger.info("开始回测 %s：最近 %s 天", symbol, days)
        threshold = _threshold_for_symbol(symbol, score_threshold, symbol_thresholds)
        allowed_entry_days = dynamic_entry_days.get(symbol)
        if allowed_entry_days:
            threshold = max(threshold, dynamic_alt_threshold)
        allowed_directions = {"做多观察"} if allowed_entry_days and dynamic_alt_long_only else None
        summary = await run_observation_backtest(
            symbol=symbol,
            days=days,
            score_threshold=threshold,
            hold_hours=hold_hours,
            fee_rate=fee_rate,
            funding_rate_8h=funding_rate_8h,
            target_rr=target_rr,
            confirm_minutes=confirm_minutes,
            expire_minutes=expire_minutes,
            min_hold_hours=min_hold_hours,
            max_hold_hours=max_hold_hours,
            dynamic_hold=dynamic_hold,
            allowed_entry_days=allowed_entry_days,
            allowed_directions=allowed_directions,
            time_stop_minutes=time_stop_minutes,
            min_progress_r=min_progress_r,
            r_trailing_enabled=r_trailing_enabled,
            trailing_trigger_pct=trailing_trigger_pct,
            trailing_lock_pct=trailing_lock_pct,
            weekly_filter=weekly_filter,
            direction_thresholds=direction_thresholds,
            conflict_threshold=conflict_threshold,
            conflict_minutes=conflict_minutes,
        )
        summaries.append(summary)
        report = summary.report_zh()
        reports.append(report)
        logger.info("\n%s", report)

    if export_trades is not None:
        out = export_backtest_trades_csv(summaries, export_trades)
        logger.info("回测交易明细已导出：%s", out)

    if notifier is not None:
        await notifier.send_text("观察型策略回测摘要\n\n" + "\n\n".join(reports))


def main() -> None:
    parser = argparse.ArgumentParser(description="加密货币观察型信号机器人")
    parser.add_argument(
        "--telegram-ping",
        action="store_true",
        help="启动后发送一条 Telegram 测试消息（使用 .env 配置）",
    )
    parser.add_argument(
        "--market-test",
        action="store_true",
        help="拉取一次 Binance 实时K线，并运行旧版 pullback_long 策略检测",
    )
    parser.add_argument(
        "--scan-market",
        action="store_true",
        help="扫描 BTC/ETH/SOL 与 24h 成交额前10合约，输出观察型入场信号",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="使用 Binance 历史合约K线回测观察型策略",
    )
    parser.add_argument(
        "--direction-eval",
        action="store_true",
        help="基于回测交易明细CSV评估方向准确率（默认1小时）",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="旧版 --market-test 的交易对，多个用英文逗号分隔",
    )
    parser.add_argument(
        "--intervals",
        default="15m,1h",
        help="旧版 --market-test 的周期，多个用英文逗号分隔",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="旧版 --market-test 每个交易对和周期拉取的K线数量",
    )
    parser.add_argument(
        "--fixed-symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="观察型扫描的固定核心交易对，多个用英文逗号分隔",
    )
    parser.add_argument(
        "--top-volume-limit",
        type=int,
        default=0,
        help="观察型扫描额外无条件加入的 24h 成交额前 N 个 USDT 合约，默认0；动态山寨由 --dynamic-alt-limit 控制",
    )
    parser.add_argument(
        "--dynamic-alt-limit",
        type=int,
        default=3,
        help="动态山寨机会池每天最多加入 N 个币，默认3；设为0关闭",
    )
    parser.add_argument(
        "--dynamic-alt-top-limit",
        type=int,
        default=30,
        help="动态山寨先从当前成交额前 N 个 USDT 合约里筛选，默认30",
    )
    parser.add_argument(
        "--dynamic-alt-lookback-days",
        type=int,
        default=7,
        help="动态山寨成交额放大对比的历史天数，默认7",
    )
    parser.add_argument(
        "--dynamic-alt-volume-ratio",
        type=float,
        default=1.5,
        help="昨日成交额至少达到过去均值的倍数，默认1.5",
    )
    parser.add_argument(
        "--dynamic-alt-min-quote-volume",
        type=float,
        default=100_000_000,
        help="动态山寨昨日估算成交额下限，默认100000000 USDT",
    )
    parser.add_argument(
        "--dynamic-alt-min-daily-move-pct",
        type=float,
        default=0.0,
        help="动态山寨昨日涨跌幅下限，默认0.0，即只允许非下跌放量进入机会池",
    )
    parser.add_argument(
        "--dynamic-alt-max-daily-move-pct",
        type=float,
        default=0.25,
        help="动态山寨昨日绝对涨跌幅上限，默认0.25，即25%%",
    )
    parser.add_argument(
        "--dynamic-alt-min-daily-range-pct",
        type=float,
        default=0.03,
        help="动态山寨昨日日内振幅下限，默认0.03，即3%%",
    )
    parser.add_argument(
        "--dynamic-alt-max-daily-range-pct",
        type=float,
        default=0.18,
        help="动态山寨昨日日内振幅上限，默认0.18，即18%%",
    )
    parser.add_argument(
        "--dynamic-alt-threshold",
        type=int,
        default=95,
        help="动态山寨推送/回测最低评分阈值，默认95",
    )
    parser.add_argument(
        "--dynamic-alt-allow-short",
        action="store_true",
        help="允许动态山寨机会池出现做空候选；默认关闭，只允许做多观察",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=90,
        help="观察型扫描的 Telegram 推送评分阈值",
    )
    parser.add_argument(
        "--symbol-thresholds",
        default="SOLUSDT:95",
        help="分币种评分阈值覆盖，例如 BTCUSDT:90,ETHUSDT:90,SOLUSDT:95",
    )
    parser.add_argument(
        "--direction-thresholds",
        default="BTCUSDT:short:98,ETHUSDT:long:96,ETHUSDT:short:98",
        help="分币种+方向阈值覆盖，例如 BTCUSDT:short:98,ETHUSDT:long:96；方向可用 long/short 或 做多/做空",
    )
    parser.add_argument(
        "--relative-rank-limit",
        type=int,
        default=3,
        help="观察型扫描只推送近6小时相对强/弱排名前 N 的币，默认3；设为0关闭",
    )
    parser.add_argument(
        "--hold-hours",
        type=float,
        default=2.0,
        help="信号基础观察/持仓时间，默认约 2 小时；动态持仓会在此基础上调整",
    )
    parser.add_argument(
        "--min-hold-hours",
        type=float,
        default=1.0,
        help="动态持仓的最短观察/持仓时间，默认约 1 小时",
    )
    parser.add_argument(
        "--max-hold-hours",
        type=float,
        default=4.0,
        help="动态持仓的最长观察/持仓时间，默认约 4 小时",
    )
    parser.add_argument(
        "--fixed-hold",
        action="store_true",
        help="关闭动态持仓，回测和扫描都固定使用 --hold-hours",
    )
    parser.add_argument(
        "--backtest-symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="回测交易对，多个用英文逗号分隔",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=30,
        help="回测最近 N 天历史数据",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0005,
        help="单边手续费率，默认 0.0005，回测会按开平双边扣除",
    )
    parser.add_argument(
        "--funding-rate-8h",
        type=float,
        default=0.0001,
        help="资金费率保守估算，默认每8小时扣除0.0001，即0.01%%；按实际持仓时长折算",
    )
    parser.add_argument(
        "--target-rr",
        type=float,
        default=2.0,
        help="观察型策略目标盈亏比，默认 2.0，即止盈1约为2R",
    )
    parser.add_argument(
        "--confirm-minutes",
        type=int,
        default=10,
        help="候选备案后，允许价格站稳触发价的确认窗口，默认10分钟",
    )
    parser.add_argument(
        "--expire-minutes",
        type=int,
        default=20,
        help="候选备案过期时间，默认20分钟",
    )
    parser.add_argument(
        "--time-stop-minutes",
        type=int,
        default=45,
        help="无进展淘汰窗口，确认后 N 分钟仍未走出有效盈利波动就提前退出，默认45分钟；设为0关闭",
    )
    parser.add_argument(
        "--min-progress-r",
        type=float,
        default=0.35,
        help="时间止损要求的最低有利波动，默认0.35R",
    )
    parser.add_argument(
        "--no-r-trailing",
        action="store_true",
        help="关闭R倍数阶梯移动止损保护",
    )
    parser.add_argument(
        "--trailing-trigger-pct",
        type=float,
        default=0.03,
        help="移动止损触发阈值，默认0.03，表示向有利方向走出3%%",
    )
    parser.add_argument(
        "--trailing-lock-pct",
        type=float,
        default=0.015,
        help="移动止损锁定幅度，默认0.015，表示触发后止损提高到锁定约1.5%%",
    )
    parser.add_argument(
        "--no-weekly-filter",
        action="store_true",
        help="关闭近一周趋势方向过滤",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=int,
        default=90,
        help="BTC 相反候选冲突过滤阈值，默认90",
    )
    parser.add_argument(
        "--conflict-minutes",
        type=int,
        default=60,
        help="回测中检查 BTC 相反信号的时间窗口，默认60分钟",
    )
    parser.add_argument(
        "--export-trades",
        nargs="?",
        const="backtest_trades.csv",
        default=None,
        help="导出回测交易明细CSV，可选指定文件名",
    )
    parser.add_argument(
        "--direction-trades-csv",
        default="backtest_trades_2025_to_now_btc_eth.csv",
        help="方向准确率评估用的回测交易明细CSV路径",
    )
    parser.add_argument(
        "--direction-horizon-minutes",
        type=int,
        default=60,
        help="方向准确率评估：入场后 N 分钟做方向判定，默认60",
    )
    parser.add_argument(
        "--direction-min-move-pct",
        type=float,
        default=0.0,
        help="方向准确率评估：|涨跌幅|<=该值(百分比) 视为平局，默认0",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="把测试或扫描结果发送到 Telegram",
    )
    parser.add_argument(
        "--paper-record",
        action="store_true",
        help="扫描达标信号时写入虚拟盘账本；通常与 --scan-market 一起使用",
    )
    parser.add_argument(
        "--symbol-cooldown-minutes",
        type=int,
        default=60,
        help="同一币种同方向冷却时间（开/平仓后不再重复记录同方向），默认60分钟；设为0关闭",
    )
    parser.add_argument(
        "--paper-sync",
        action="store_true",
        help="用最新行情更新虚拟盘账本里的待触发和持仓记录",
    )
    parser.add_argument(
        "--paper-summary",
        action="store_true",
        help="输出虚拟盘账本摘要",
    )
    parser.add_argument(
        "--paper-path",
        default="paper_trades.json",
        help="虚拟盘账本文件路径，默认 paper_trades.json",
    )
    parser.add_argument(
        "--paper-entry-mode",
        default="immediate",
        help="虚拟盘入场模式：immediate(信号即按当前价入场) 或 confirm(等待站稳触发价)，默认 immediate",
    )
    parser.add_argument(
        "--paper-export",
        default="paper_trades.csv",
        help="虚拟盘导出文件路径（.csv 或 .xlsx），默认 paper_trades.csv",
    )
    parser.add_argument(
        "--paper-export-every-hours",
        type=int,
        default=24,
        help="常驻运行时每隔N小时导出一次（0表示每轮循环都导出），默认24",
    )
    parser.add_argument(
        "--paper-snapshot-dir",
        default="paper_snapshots",
        help="最终快照CSV输出目录（断网/异常/手动停止时落盘），默认 paper_snapshots",
    )
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="常驻运行：循环扫描市场、同步虚拟盘、导出表格（Ctrl+C 停止）",
    )
    parser.add_argument(
        "--loop-minutes",
        type=int,
        default=5,
        help="常驻运行循环间隔分钟数，默认5",
    )
    parser.add_argument(
        "--market-summary-every-scans",
        type=int,
        default=12,
        help="常驻运行：每N轮扫描发送一次市场总结（0关闭）。默认12（当 --loop-minutes=5 时约等于每小时一次）",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.validate()
    configure_logging(settings.log_level)

    logger.info("配置加载完成；日志级别=%s", settings.log_level)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    symbol_thresholds = _parse_symbol_thresholds(args.symbol_thresholds)
    direction_thresholds = _parse_direction_thresholds(args.direction_thresholds)

    if args.telegram_ping:
        asyncio.run(_optional_ping(notifier))
        logger.info("Telegram 测试消息已发送")

    if args.market_test:
        symbols = _parse_csv(args.symbols)
        intervals = _parse_csv(args.intervals)
        if not symbols:
            raise ValueError("--symbols 至少需要包含一个交易对")
        if not intervals:
            raise ValueError("--intervals 至少需要包含一个周期")
        if args.limit < 120:
            raise ValueError("--limit 必须大于等于 120，当前策略需要足够的历史K线")
        asyncio.run(
            _market_test(
                symbols=symbols,
                intervals=intervals,
                limit=args.limit,
                notifier=notifier if args.notify else None,
            )
        )
        logger.info("市场检测完成")

    if args.scan_market:
        if args.top_volume_limit < 0:
            raise ValueError("--top-volume-limit 不能小于 0")
        if args.dynamic_alt_limit < 0 or args.dynamic_alt_top_limit < 0:
            raise ValueError("--dynamic-alt-limit 与 --dynamic-alt-top-limit 不能小于 0")
        if args.dynamic_alt_lookback_days < 2:
            raise ValueError("--dynamic-alt-lookback-days 至少为 2")
        if args.dynamic_alt_volume_ratio <= 0 or args.dynamic_alt_min_quote_volume < 0:
            raise ValueError("--dynamic-alt-volume-ratio 必须大于0，--dynamic-alt-min-quote-volume 不能小于0")
        if args.dynamic_alt_min_daily_move_pct > args.dynamic_alt_max_daily_move_pct:
            raise ValueError("--dynamic-alt-min-daily-move-pct 不能大于最大涨跌幅")
        if args.dynamic_alt_min_daily_range_pct < 0 or args.dynamic_alt_min_daily_range_pct > args.dynamic_alt_max_daily_range_pct:
            raise ValueError("--dynamic-alt-min-daily-range-pct 不能小于0，且不能大于最大振幅")
        if not 0 <= args.dynamic_alt_threshold <= 100:
            raise ValueError("--dynamic-alt-threshold 必须在 0 到 100 之间")
        if not 0 <= args.score_threshold <= 100:
            raise ValueError("--score-threshold 必须在 0 到 100 之间")
        if args.min_hold_hours <= 0 or args.max_hold_hours <= 0 or args.min_hold_hours > args.max_hold_hours:
            raise ValueError("--min-hold-hours 与 --max-hold-hours 必须大于0，且最短时间不能大于最长时间")
        if args.relative_rank_limit < 0:
            raise ValueError("--relative-rank-limit 不能小于 0")
        fixed_symbols = _parse_csv(args.fixed_symbols)
        asyncio.run(
            _scan_market(
                fixed_symbols=fixed_symbols,
                top_volume_limit=args.top_volume_limit,
                score_threshold=args.score_threshold,
                expected_hold_hours=args.hold_hours,
                min_hold_hours=args.min_hold_hours,
                max_hold_hours=args.max_hold_hours,
                dynamic_hold=not args.fixed_hold,
                target_rr=args.target_rr,
                expire_minutes=args.expire_minutes,
                symbol_thresholds=symbol_thresholds,
                direction_thresholds=direction_thresholds,
                dynamic_alt_limit=args.dynamic_alt_limit,
                dynamic_alt_top_limit=args.dynamic_alt_top_limit,
                dynamic_alt_lookback_days=args.dynamic_alt_lookback_days,
                dynamic_alt_volume_ratio=args.dynamic_alt_volume_ratio,
                dynamic_alt_min_quote_volume=args.dynamic_alt_min_quote_volume,
                dynamic_alt_min_daily_move_pct=args.dynamic_alt_min_daily_move_pct,
                dynamic_alt_max_daily_move_pct=args.dynamic_alt_max_daily_move_pct,
                dynamic_alt_min_daily_range_pct=args.dynamic_alt_min_daily_range_pct,
                dynamic_alt_max_daily_range_pct=args.dynamic_alt_max_daily_range_pct,
                dynamic_alt_threshold=args.dynamic_alt_threshold,
                dynamic_alt_long_only=not args.dynamic_alt_allow_short,
                conflict_threshold=args.conflict_threshold,
                relative_rank_limit=args.relative_rank_limit,
                weekly_filter=not args.no_weekly_filter,
                paper_record=args.paper_record,
                paper_path=args.paper_path,
                confirm_minutes=args.confirm_minutes,
                paper_entry_mode=args.paper_entry_mode,
                symbol_cooldown_minutes=args.symbol_cooldown_minutes,
                notifier=notifier if args.notify else None,
            )
        )
        logger.info("观察型市场扫描完成")

    if args.backtest:
        symbols = _parse_csv(args.backtest_symbols)
        if not symbols:
            raise ValueError("--backtest-symbols 至少需要包含一个交易对")
        if args.backtest_days < 1:
            raise ValueError("--backtest-days 必须大于 0")
        if args.dynamic_alt_limit < 0 or args.dynamic_alt_top_limit < 0:
            raise ValueError("--dynamic-alt-limit 与 --dynamic-alt-top-limit 不能小于 0")
        if args.dynamic_alt_lookback_days < 2:
            raise ValueError("--dynamic-alt-lookback-days 至少为 2")
        if args.dynamic_alt_volume_ratio <= 0 or args.dynamic_alt_min_quote_volume < 0:
            raise ValueError("--dynamic-alt-volume-ratio 必须大于0，--dynamic-alt-min-quote-volume 不能小于0")
        if args.dynamic_alt_min_daily_move_pct > args.dynamic_alt_max_daily_move_pct:
            raise ValueError("--dynamic-alt-min-daily-move-pct 不能大于最大涨跌幅")
        if args.dynamic_alt_min_daily_range_pct < 0 or args.dynamic_alt_min_daily_range_pct > args.dynamic_alt_max_daily_range_pct:
            raise ValueError("--dynamic-alt-min-daily-range-pct 不能小于0，且不能大于最大振幅")
        if not 0 <= args.dynamic_alt_threshold <= 100:
            raise ValueError("--dynamic-alt-threshold 必须在 0 到 100 之间")
        if args.min_hold_hours <= 0 or args.max_hold_hours <= 0 or args.min_hold_hours > args.max_hold_hours:
            raise ValueError("--min-hold-hours 与 --max-hold-hours 必须大于0，且最短时间不能大于最长时间")
        if args.fee_rate < 0 or args.funding_rate_8h < 0:
            raise ValueError("--fee-rate 与 --funding-rate-8h 不能小于 0")
        asyncio.run(
            _backtest(
                symbols=symbols,
                days=args.backtest_days,
                score_threshold=args.score_threshold,
                hold_hours=args.hold_hours,
                fee_rate=args.fee_rate,
                funding_rate_8h=args.funding_rate_8h,
                target_rr=args.target_rr,
                confirm_minutes=args.confirm_minutes,
                expire_minutes=args.expire_minutes,
                min_hold_hours=args.min_hold_hours,
                max_hold_hours=args.max_hold_hours,
                dynamic_hold=not args.fixed_hold,
                time_stop_minutes=args.time_stop_minutes,
                min_progress_r=args.min_progress_r,
                r_trailing_enabled=not args.no_r_trailing,
                trailing_trigger_pct=args.trailing_trigger_pct,
                trailing_lock_pct=args.trailing_lock_pct,
                weekly_filter=not args.no_weekly_filter,
                symbol_thresholds=symbol_thresholds,
                direction_thresholds=direction_thresholds,
                dynamic_alt_limit=args.dynamic_alt_limit,
                dynamic_alt_top_limit=args.dynamic_alt_top_limit,
                dynamic_alt_lookback_days=args.dynamic_alt_lookback_days,
                dynamic_alt_volume_ratio=args.dynamic_alt_volume_ratio,
                dynamic_alt_min_quote_volume=args.dynamic_alt_min_quote_volume,
                dynamic_alt_min_daily_move_pct=args.dynamic_alt_min_daily_move_pct,
                dynamic_alt_max_daily_move_pct=args.dynamic_alt_max_daily_move_pct,
                dynamic_alt_min_daily_range_pct=args.dynamic_alt_min_daily_range_pct,
                dynamic_alt_max_daily_range_pct=args.dynamic_alt_max_daily_range_pct,
                dynamic_alt_threshold=args.dynamic_alt_threshold,
                dynamic_alt_long_only=not args.dynamic_alt_allow_short,
                conflict_threshold=args.conflict_threshold,
                conflict_minutes=args.conflict_minutes,
                export_trades=args.export_trades,
                notifier=notifier if args.notify else None,
            )
        )
        logger.info("观察型策略回测完成")

    if args.direction_eval:
        asyncio.run(
            _direction_accuracy_1h(
                trades_csv=args.direction_trades_csv,
                horizon_minutes=args.direction_horizon_minutes,
                min_move_pct=args.direction_min_move_pct,
            )
        )
        logger.info("方向准确率评估完成")

    if args.paper_sync:
        if args.fee_rate < 0 or args.funding_rate_8h < 0:
            raise ValueError("--fee-rate 与 --funding-rate-8h 不能小于 0")
        if args.time_stop_minutes < 0 or args.min_progress_r < 0:
            raise ValueError("--time-stop-minutes 与 --min-progress-r 不能小于 0")
        result = asyncio.run(
            sync_paper_trades(
                args.paper_path,
                fee_rate=args.fee_rate,
                funding_rate_8h=args.funding_rate_8h,
                time_stop_minutes=args.time_stop_minutes,
                min_progress_r=args.min_progress_r,
                r_trailing_enabled=not args.no_r_trailing,
                trailing_trigger_pct=args.trailing_trigger_pct,
                trailing_lock_pct=args.trailing_lock_pct,
            )
        )
        report_lines = [f"虚拟盘同步完成：更新 {result.changed_count} 条记录"]
        report_lines.extend(result.errors)
        report_lines.extend(["", paper_summary_zh(result.trades)])
        report = "\n".join(report_lines)
        logger.info("\n%s", report)
        try:
            exported = export_paper_trades(args.paper_export, result.trades)
            logger.info("虚拟盘已导出：%s", exported)
        except Exception as exc:
            logger.warning("虚拟盘导出失败：%s", exc)
        if args.notify:
            asyncio.run(notifier.send_text(report))

    if args.paper_summary and not args.paper_sync:
        report = paper_summary_zh(load_paper_trades(args.paper_path))
        logger.info("\n%s", report)
        if args.notify:
            asyncio.run(notifier.send_text(report))

    if args.run_live:
        if args.loop_minutes < 1:
            raise ValueError("--loop-minutes 必须大于 0")
        try:
            asyncio.run(
                _run_live_loop(
                    fixed_symbols=_parse_csv(args.fixed_symbols),
                    top_volume_limit=args.top_volume_limit,
                    score_threshold=args.score_threshold,
                    expected_hold_hours=args.hold_hours,
                    min_hold_hours=args.min_hold_hours,
                    max_hold_hours=args.max_hold_hours,
                    dynamic_hold=not args.fixed_hold,
                    target_rr=args.target_rr,
                    expire_minutes=args.expire_minutes,
                    symbol_thresholds=symbol_thresholds,
                    direction_thresholds=direction_thresholds,
                    dynamic_alt_limit=args.dynamic_alt_limit,
                    dynamic_alt_top_limit=args.dynamic_alt_top_limit,
                    dynamic_alt_lookback_days=args.dynamic_alt_lookback_days,
                    dynamic_alt_volume_ratio=args.dynamic_alt_volume_ratio,
                    dynamic_alt_min_quote_volume=args.dynamic_alt_min_quote_volume,
                    dynamic_alt_min_daily_move_pct=args.dynamic_alt_min_daily_move_pct,
                    dynamic_alt_max_daily_move_pct=args.dynamic_alt_max_daily_move_pct,
                    dynamic_alt_min_daily_range_pct=args.dynamic_alt_min_daily_range_pct,
                    dynamic_alt_max_daily_range_pct=args.dynamic_alt_max_daily_range_pct,
                    dynamic_alt_threshold=args.dynamic_alt_threshold,
                    dynamic_alt_long_only=not args.dynamic_alt_allow_short,
                    conflict_threshold=args.conflict_threshold,
                    relative_rank_limit=args.relative_rank_limit,
                    weekly_filter=not args.no_weekly_filter,
                    paper_path=args.paper_path,
                    paper_entry_mode=args.paper_entry_mode,
                    confirm_minutes=args.confirm_minutes,
                    paper_export_path=args.paper_export,
                    paper_export_every_hours=args.paper_export_every_hours,
                    paper_snapshot_dir=args.paper_snapshot_dir,
                    loop_minutes=args.loop_minutes,
                    market_summary_every_scans=args.market_summary_every_scans,
                    symbol_cooldown_minutes=args.symbol_cooldown_minutes,
                    fee_rate=args.fee_rate,
                    funding_rate_8h=args.funding_rate_8h,
                    time_stop_minutes=args.time_stop_minutes,
                    min_progress_r=args.min_progress_r,
                    r_trailing_enabled=not args.no_r_trailing,
                    trailing_trigger_pct=args.trailing_trigger_pct,
                    trailing_lock_pct=args.trailing_lock_pct,
                    notifier=notifier if args.notify else None,
                )
            )
        except KeyboardInterrupt:
            try:
                now_utc = datetime.now(UTC)
                snap_dir = Path(args.paper_snapshot_dir)
                snap_dir.mkdir(parents=True, exist_ok=True)
                exported = export_paper_trades(
                    snap_dir / Path(_timestamped_export_path(Path(args.paper_export).name, now_utc, "stopped")),
                    load_paper_trades(args.paper_path),
                )
                logger.info("已停止运行；虚拟盘最终快照已导出：%s", exported)
            except Exception as exc:
                logger.warning("停止时导出最终快照失败：%s", exc)

    if (
        not args.telegram_ping
        and not args.market_test
        and not args.scan_market
        and not args.backtest
        and not args.direction_eval
        and not args.paper_sync
        and not args.paper_summary
        and not args.run_live
    ):
        logger.info("Telegram 通知器已就绪（可使用 --telegram-ping 测试）")


if __name__ == "__main__":
    main()
