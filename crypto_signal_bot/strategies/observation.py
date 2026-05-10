from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from crypto_signal_bot.indicators import closes, ema, ema_slope, ma25, ma99, volume_ma
from crypto_signal_bot.models import Candle
from crypto_signal_bot.strategies.market_regime import MarketRegimeResult, detect_market_regime

MIN_TREND_BARS = 120
MIN_ENTRY_BARS = 120
SIDEWAYS_EXCEPTION_SCORE = 95
SIDEWAYS_SCORE_PENALTY = 5
MIN_POSITION_SPACE_R = 1.2
GOOD_POSITION_SPACE_R = 2.0
SHORT_TREND_EMA_LOOKBACK = 1
MIN_VOLUME_SPIKE_RATIO = 1.3

# 15m bars: we use a longer range window to reduce false breakouts from very short consolidations.
ENTRY_RANGE_LOOKBACK_BARS = 32  # ~8 hours on 15m
ENTRY_STOP_LOOKBACK_BARS = 16  # ~4 hours on 15m

# If long/short scores are too close, it usually means the market is choppy or flipping.
# Filtering these "low-confidence" states improves directional accuracy.
MIN_DIRECTION_SCORE_GAP = 6

# Efficiency ratio (Kaufman): net change / sum(|delta|). Signed in [-1, 1].
TREND_EFFICIENCY_LOOKBACK_BARS = 24  # 24x 1h bars ~= 1 day
MIN_TREND_EFFICIENCY_ABS = 0.12


@dataclass(frozen=True)
class MarketRegime:
    direction: str
    allows_long: bool
    allows_short: bool
    detail: str


@dataclass(frozen=True)
class WeeklyTrend:
    direction: str
    allows_long: bool
    allows_short: bool
    detail: str


@dataclass(frozen=True)
class HoldTimePlan:
    hold_hours: float
    stage: str
    strength: str
    detail: str


@dataclass(frozen=True)
class PositionContext:
    score_adjustment: int
    detail: str


@dataclass(frozen=True)
class ObservationCandidate:
    symbol: str
    direction: str
    score: int
    level: str
    current_price: float
    trigger_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    target_rr: float
    expires_after_minutes: int
    expected_hold_hours: float
    reasons: tuple[str, ...]
    # New scoring + hard filters (hard filters gate paper-trade entry; soft score only ranks).
    legacy_score: int = 0
    hard_filter_passed: bool = False
    failed_hard_filters: tuple[str, ...] = ()
    raw_score: float = 0.0
    final_score: int = 0
    score_components_json: str = ""
    # Market regime filter (independent of score; used for trade gating)
    market_regime: str = "unknown"
    market_regime_confidence: int = 0
    no_trade_zone: bool = False
    market_regime_reasons: tuple[str, ...] = ()

    @property
    def signal_grade(self) -> str:
        if self.score >= 95:
            return "A级：可重点观察"
        if self.score >= 90:
            return "B级：只做观察，等待人工确认"
        if self.score >= 80:
            return "C级：记录为主，不建议实盘"
        return "D级：暂不关注"

    @property
    def action_hint(self) -> str:
        if self.score >= 95:
            return "只有人工确认清单大部分通过时，才考虑小资金低倍测试。"
        if self.score >= 90:
            return "先备案观察，不建议直接开仓；必须等待触发后的二次确认。"
        return "只记录，不作为实盘测试候选。"

    def summary_line(self) -> str:
        return (
            f"{self.symbol}：{self.direction}候选，评分={self.score}，"
            f"现价={self.current_price:g}，触发价={self.trigger_price:g}，"
            f"止损={self.stop_loss:g}，止盈1={self.take_profit_1:g}"
        )

    def to_telegram_message(self) -> str:
        reason_text = "\n".join(f"{i}. {r}" for i, r in enumerate(self.reasons, start=1))
        checklist_text = "\n".join(f"{i}. {r}" for i, r in enumerate(_manual_confirmation_checklist(self), start=1))
        followup_text = "\n".join(f"{i}. {r}" for i, r in enumerate(_post_trigger_confirmation_plan(self), start=1))
        header = "观察信号"
        hard_line = ""
        if self.direction in {"做多观察", "做空观察"}:
            if self.hard_filter_passed:
                header = "硬过滤通过，等待触发"
                hard_line = "硬过滤：通过（已进入 pending，等待触发确认后入场）\n"
            elif self.failed_hard_filters:
                header = "硬过滤失败，仅记录，不交易"
                hard_line = f"硬过滤：失败（{', '.join(self.failed_hard_filters)}）\n"
        return (
            f"{header}：{self.symbol} {self.direction}\n\n"
            f"信号级别：{self.level} / {self.signal_grade}\n"
            f"综合评分：{self.score} / 100（legacy={self.legacy_score}）\n"
            f"{hard_line}"
            f"处理建议：{self.action_hint}\n"
            f"当前价格：{self.current_price:g}\n"
            f"触发条件：未来短时间内有效站稳 {self.trigger_price:g}\n"
            f"止损参考：{self.stop_loss:g}\n"
            f"止盈1：{self.take_profit_1:g}\n"
            f"止盈2：{self.take_profit_2:g}\n"
            f"目标盈亏比：约 {self.target_rr:g}:1\n"
            f"备案有效期：约 {self.expires_after_minutes} 分钟\n"
            f"预计观察/持仓：约 {self.expected_hold_hours:g} 小时\n\n"
            f"备案理由：\n{reason_text}\n\n"
            f"人工确认清单：\n{checklist_text}\n\n"
            f"触发后二次确认：\n{followup_text}\n\n"
            "风险提示：这是观察型备案，不是自动下单建议。确认失败就应该作废，"
            "低倍合约也需要提前设好单笔最大亏损。"
        )


@dataclass(frozen=True)
class ObservationSignal:
    symbol: str
    direction: str
    score: int
    level: str
    current_price: float
    reference_entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    expected_hold_hours: float
    reasons: tuple[str, ...]

    def summary_line(self) -> str:
        return (
            f"{self.symbol}：{self.direction}，评分={self.score}，"
            f"现价={self.current_price:g}，止损={self.stop_loss:g}，"
            f"止盈1={self.take_profit_1:g}，止盈2={self.take_profit_2:g}"
        )

    def to_telegram_message(self) -> str:
        reason_text = "\n".join(f"{i}. {r}" for i, r in enumerate(self.reasons, start=1))
        return (
            f"交易观察：{self.symbol} {self.direction}\n\n"
            f"信号级别：{self.level}\n"
            f"综合评分：{self.score} / 100\n"
            f"当前价格：{self.current_price:g}\n"
            f"参考入场：{self.reference_entry:g}\n"
            f"止损参考：{self.stop_loss:g}\n"
            f"止盈1：{self.take_profit_1:g}\n"
            f"止盈2：{self.take_profit_2:g}\n"
            f"预计观察/持仓：约 {self.expected_hold_hours:g} 小时\n\n"
            f"信号理由：\n{reason_text}\n\n"
            "风险提示：这是观察信号，不是自动下单建议。低倍合约也会放大亏损，"
            "建议只用小资金测试，并提前设好单笔最大亏损。"
        )


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _level(score: int) -> str:
    if score >= 85:
        return "强观察"
    if score >= 75:
        return "可观察"
    if score >= 65:
        return "弱观察"
    return "暂不关注"


def _manual_confirmation_checklist(candidate: ObservationCandidate) -> tuple[str, ...]:
    return (
        "BTC 和 ETH 最好与本单方向一致；若两者明显相反，本单降级为只观察。",
        "检查4h关键支撑/压力：目标空间要足够，不能刚入场就贴近阻力或支撑。",
        "做多优先选择相对 BTC 更强的币；做空优先选择相对 BTC 更弱的币。",
        "触发价不能离现价太远，也不能已经远离结构区后追进去。",
        "确认未来30-45分钟能看一眼走势；不能看盘就不做实盘测试。",
        "单笔亏损先固定上限，低倍也要提前设置止损。",
    )


def _post_trigger_confirmation_plan(candidate: ObservationCandidate) -> tuple[str, ...]:
    return (
        "触发后10-15分钟需要继续沿信号方向推进；只碰一下触发价不算强确认。",
        "入场后45分钟仍未走出约0.35R，信号视为衰减，应主动放弃或减仓。",
        "走到0.8R后开始收窄风险；走到1R后先保护本金或半仓止盈。",
        "走到1.5R以后，剩余仓位止损逐步抬到盈利区，不再让盈利单变大亏。",
    )


def _avg_true_range(candles: Sequence[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    ranges: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        ranges.append(max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close)))
    return sum(ranges) / period


def _rsi(candles: Sequence[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(len(candles) - period, len(candles)):
        change = candles[i].close - candles[i - 1].close
        if change >= 0:
            gains += change
        else:
            losses += abs(change)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(candles: Sequence[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None

    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        current = candles[i]
        prev = candles[i - 1]
        up_move = current.high - prev.high
        down_move = prev.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(current.high - current.low, abs(current.high - prev.close), abs(current.low - prev.close)))

    tr_sum = sum(trs)
    if tr_sum <= 0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm) / tr_sum
    minus_di = 100.0 * sum(minus_dm) / tr_sum
    denom = plus_di + minus_di
    if denom <= 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / denom


def _ma25_slope(candles: Sequence[Candle]) -> float:
    if len(candles) < 50:
        return 0.0
    prev = sum(c.close for c in candles[-50:-25]) / 25
    now = sum(c.close for c in candles[-25:]) / 25
    return now - prev


def _short_term_trend_ok(
    candles: Sequence[Candle],
    direction: str,
    *,
    reference_price: float | None = None,
) -> tuple[bool, str]:
    closes_ = closes(candles)
    e20 = ema(closes_, 20)
    e50 = ema(closes_, 50)
    s20 = ema_slope(closes_, 20, SHORT_TREND_EMA_LOOKBACK)
    s50 = ema_slope(closes_, 50, SHORT_TREND_EMA_LOOKBACK)
    if None in {e20, e50, s20, s50}:
        return False, "短周期EMA指标未就绪"
    last = candles[-1].close
    price = reference_price if reference_price is not None else last
    # EMA 斜率在拐点附近容易出现很小的负值，这里允许轻微回撤，只要结构仍然向上。
    eps20 = (e20 or 0.0) * 0.00012
    eps50 = (e50 or 0.0) * 0.00008
    # EMA20 与 EMA50 经常在临界点附近来回贴合；给一点点容错，避免把“刚要走出来”的行情过滤掉。
    align_eps = (e50 or 0.0) * 0.0006
    if direction == "做多观察":
        align_ok = (e20 + align_eps) >= e50
        ok = price > e20 and align_ok and s20 > -eps20 and s50 > -eps50
        detail = (
            f"EMA20/50结构：price={price:g} close={last:g} EMA20={e20:.4g} EMA50={e50:.4g} "
            f"slope20={s20:.4g} slope50={s50:.4g}"
        )
        return ok, detail
    if direction == "做空观察":
        align_ok = (e20 - align_eps) <= e50
        ok = price < e20 and align_ok and s20 < eps20 and s50 < eps50
        detail = (
            f"EMA20/50结构：price={price:g} close={last:g} EMA20={e20:.4g} EMA50={e50:.4g} "
            f"slope20={s20:.4g} slope50={s50:.4g}"
        )
        return ok, detail
    return False, "方向非法"


def _volume_spike_ok(entry_candles: Sequence[Candle], ratio: float = MIN_VOLUME_SPIKE_RATIO) -> tuple[bool, str]:
    vma = volume_ma(entry_candles, 20)
    if vma is None or vma <= 0:
        return False, "成交量均值未就绪"
    last_v = entry_candles[-1].volume
    multiple = last_v / vma
    ok = multiple >= ratio
    return ok, f"成交量放大：{multiple:.2f}x（阈值 {ratio:.2f}x）"


def _pct_change(candles: Sequence[Candle], lookback: int) -> float | None:
    if len(candles) <= lookback:
        return None
    old = candles[-lookback - 1].close
    if old <= 0:
        return None
    return (candles[-1].close - old) / old


def _efficiency_ratio(candles: Sequence[Candle], lookback: int) -> float | None:
    """
    Kaufman Efficiency Ratio (signed):
      ER = (price[t] - price[t-lookback]) / sum(|price[i] - price[i-1]|), i in (t-lookback+1..t)

    Returns values in [-1, 1]. Magnitude measures trend efficiency; sign gives direction.
    """
    if lookback <= 0 or len(candles) <= lookback:
        return None
    start = candles[-lookback - 1].close
    end = candles[-1].close
    denom = 0.0
    for i in range(len(candles) - lookback, len(candles)):
        denom += abs(candles[i].close - candles[i - 1].close)
    if denom <= 0:
        return 0.0
    return (end - start) / denom


def _aggregate_candles(candles: Sequence[Candle], bars_per_candle: int) -> list[Candle]:
    if bars_per_candle <= 1:
        return list(candles)
    out: list[Candle] = []
    for start in range(0, len(candles), bars_per_candle):
        chunk = list(candles[start : start + bars_per_candle])
        if len(chunk) < bars_per_candle:
            continue
        out.append(
            Candle(
                open_time=chunk[0].open_time,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
                close_time=chunk[-1].close_time,
                is_closed=all(c.is_closed for c in chunk),
            )
        )
    return out


def _position_context(
    *,
    direction: str,
    trend_candles: Sequence[Candle],
    trigger_price: float,
    stop_loss: float,
) -> PositionContext:
    if direction not in {"做多观察", "做空观察"} or len(trend_candles) < 96:
        return PositionContext(0, "位置指标历史不足，暂不额外加减分。")

    is_long = direction == "做多观察"
    risk = max(abs(trigger_price - stop_loss), trigger_price * 0.001)
    h4 = _aggregate_candles(trend_candles[-120:], 4)
    if len(h4) < 16:
        return PositionContext(0, "4h位置指标未就绪，暂不额外加减分。")

    recent_h4 = h4[-24:-1] if len(h4) >= 25 else h4[:-1]
    if not recent_h4:
        return PositionContext(0, "4h位置指标未就绪，暂不额外加减分。")

    trend_ma25 = ma25(trend_candles)
    trend_atr = _avg_true_range(trend_candles, 14)
    adjustment = 0
    details: list[str] = []

    if is_long:
        resistance = max(c.high for c in recent_h4)
        if resistance > trigger_price:
            space_r = (resistance - trigger_price) / risk
            if space_r < 0.8:
                adjustment -= 14
                details.append(f"4h上方压力距离仅约 {space_r:.2f}R，做多容易刚入场就遇阻。")
            elif space_r < MIN_POSITION_SPACE_R:
                adjustment -= 8
                details.append(f"4h上方压力距离约 {space_r:.2f}R，目标空间偏紧。")
            elif space_r >= GOOD_POSITION_SPACE_R:
                details.append(f"4h上方空间约 {space_r:.2f}R，位置相对宽松。")
            else:
                details.append(f"4h上方空间约 {space_r:.2f}R，位置中性。")
        else:
            details.append("触发价已接近或高于近期4h压力，若能站稳，属于有效突破位置。")
    else:
        support = min(c.low for c in recent_h4)
        if support < trigger_price:
            space_r = (trigger_price - support) / risk
            if space_r < 0.8:
                adjustment -= 14
                details.append(f"4h下方支撑距离仅约 {space_r:.2f}R，做空容易刚入场就遇支撑。")
            elif space_r < MIN_POSITION_SPACE_R:
                adjustment -= 8
                details.append(f"4h下方支撑距离约 {space_r:.2f}R，目标空间偏紧。")
            elif space_r >= GOOD_POSITION_SPACE_R:
                details.append(f"4h下方空间约 {space_r:.2f}R，位置相对宽松。")
            else:
                details.append(f"4h下方空间约 {space_r:.2f}R，位置中性。")
        else:
            details.append("触发价已接近或低于近期4h支撑，若能跌破，属于有效破位位置。")

    if trend_ma25 is not None and trend_atr is not None and trend_atr > 0:
        distance_atr = abs(trigger_price - trend_ma25) / trend_atr
        extended = trigger_price > trend_ma25 if is_long else trigger_price < trend_ma25
        if extended and distance_atr >= 3.0:
            adjustment -= 8
            details.append(f"触发价距离1h MA25约 {distance_atr:.1f} ATR，追价风险偏高。")
        elif extended and distance_atr <= 1.8:
            details.append(f"触发价距离1h MA25约 {distance_atr:.1f} ATR，位置不算过度追价。")

    return PositionContext(adjustment, " ".join(details) if details else "位置中性，暂不额外加减分。")


def _directional_pct_change(candles: Sequence[Candle], lookback: int, direction: str) -> float | None:
    change = _pct_change(candles, lookback)
    if change is None:
        return None
    return change if direction == "做多观察" else -change


def _round_quarter_hour(hours: float) -> float:
    return round(hours * 4) / 4


def _trend_age_bars(candles: Sequence[Candle], direction: str, max_lookback: int = 96) -> int:
    if len(candles) < 99:
        return 0

    is_long = direction == "做多观察"
    age = 0
    start = max(99, len(candles) - max_lookback)
    for end in range(len(candles), start - 1, -1):
        window = candles[:end]
        m25 = ma25(window)
        m99 = ma99(window)
        if m25 is None or m99 is None:
            break
        close = window[-1].close
        slope = _ma25_slope(window)
        aligned = (
            close >= m25 and m25 >= m99 and slope > 0
            if is_long
            else close <= m25 and m25 <= m99 and slope < 0
        )
        if not aligned:
            break
        age += 1
    return age


def estimate_hold_time_plan(
    *,
    direction: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    base_hold_hours: float = 2.0,
    min_hold_hours: float = 1.0,
    max_hold_hours: float = 4.0,
    dynamic_hold: bool = True,
) -> HoldTimePlan:
    base = max(0.25, base_hold_hours)
    lower = max(0.25, min(min_hold_hours, max_hold_hours))
    upper = max(lower, max(min_hold_hours, max_hold_hours))
    fixed_hours = _round_quarter_hour(max(lower, min(upper, base)))
    if not dynamic_hold or direction not in {"做多观察", "做空观察"}:
        return HoldTimePlan(fixed_hours, "固定", "中性", f"固定观察/持仓时间：约 {fixed_hours:g} 小时。")

    trend_ma25 = ma25(trend_candles)
    trend_ma99 = ma99(trend_candles)
    trend_atr = _avg_true_range(trend_candles, 14)
    trend_rsi = _rsi(trend_candles, 14)
    trend_adx = _adx(trend_candles, 14)
    entry_adx = _adx(entry_candles, 14)
    if None in {trend_ma25, trend_ma99, trend_atr, trend_rsi, trend_adx, entry_adx}:
        return HoldTimePlan(fixed_hours, "中性", "中性", f"持仓时间指标未就绪，暂按基础 {fixed_hours:g} 小时处理。")

    assert trend_ma25 is not None
    assert trend_ma99 is not None
    assert trend_atr is not None
    assert trend_rsi is not None
    assert trend_adx is not None
    assert entry_adx is not None

    is_long = direction == "做多观察"
    trend_last = trend_candles[-1]
    slope = _ma25_slope(trend_candles)
    slope_ok = slope > 0 if is_long else slope < 0
    age = _trend_age_bars(trend_candles, direction)
    distance_atr = abs(trend_last.close - trend_ma25) / trend_atr if trend_atr > 0 else 0.0
    move_24h = _directional_pct_change(trend_candles, 24, direction) or 0.0
    move_week = _directional_pct_change(trend_candles, 168, direction) or 0.0

    too_hot = trend_rsi >= 72 if is_long else trend_rsi <= 28
    overextended = distance_atr >= 2.8 or (move_24h >= 0.055 and age >= 24) or (move_week >= 0.14 and age >= 48)

    if age <= 0:
        stage = "趋势未确认"
    elif age <= 24 and not overextended:
        stage = "趋势初段"
    elif overextended or age >= 72 or too_hot:
        stage = "趋势末段/过热"
    else:
        stage = "趋势中段"

    if trend_adx >= 25 and entry_adx >= 18 and slope_ok:
        strength = "强"
    elif trend_adx >= 18 or entry_adx >= 18:
        strength = "中"
    else:
        strength = "弱"

    hours = base
    if stage == "趋势初段":
        if strength == "强":
            hours += 1.0
        elif strength == "中":
            hours += 0.5
    elif stage == "趋势中段":
        if strength == "强":
            hours += 0.5
        elif strength == "弱":
            hours -= 0.25
    elif stage == "趋势末段/过热":
        hours -= 0.75 if strength != "弱" else 1.0
    else:
        hours -= 0.5

    if too_hot:
        hours -= 0.5
    if distance_atr >= 3.5:
        hours -= 0.25

    hold_hours = _round_quarter_hour(max(lower, min(upper, hours)))
    detail = (
        f"{strength}趋势，{stage}：1h ADX={trend_adx:.1f}，15m ADX={entry_adx:.1f}，"
        f"趋势连续约 {age} 根1hK，价格距离MA25约 {distance_atr:.1f} ATR，"
        f"近24小时顺势幅度={move_24h * 100:.2f}%，预计观察/持仓调整为 {hold_hours:g} 小时。"
    )
    return HoldTimePlan(hold_hours, stage, strength, detail)


def classify_market_regime(candles: Sequence[Candle]) -> MarketRegime:
    if len(candles) < MIN_TREND_BARS:
        return MarketRegime("中性", True, True, "BTC 1h 历史K线不足，暂按中性处理。")

    last = candles[-1]
    m25 = ma25(candles)
    m99 = ma99(candles)
    rsi = _rsi(candles, 14)
    adx = _adx(candles, 14)
    slope = _ma25_slope(candles)
    if m25 is None or m99 is None or rsi is None or adx is None:
        return MarketRegime("中性", True, True, "BTC 1h 指标未就绪，暂按中性处理。")

    strong_bear = last.close < m99 and m25 < m99 and slope < 0 and rsi < 48
    fast_bear = last.close < m25 and slope < 0 and rsi < 42 and adx >= 18
    strong_bull = last.close > m99 and m25 > m99 and slope > 0 and rsi > 52
    fast_bull = last.close > m25 and slope > 0 and rsi > 58 and adx >= 18

    if strong_bear or fast_bear:
        return MarketRegime(
            "偏弱",
            False,
            True,
            f"BTC 1h 偏弱：价格={last.close:g}，MA25={m25:g}，MA99={m99:g}，RSI={rsi:.1f}，ADX={adx:.1f}。",
        )
    if strong_bull or fast_bull:
        return MarketRegime(
            "偏强",
            True,
            False,
            f"BTC 1h 偏强：价格={last.close:g}，MA25={m25:g}，MA99={m99:g}，RSI={rsi:.1f}，ADX={adx:.1f}。",
        )
    return MarketRegime(
        "震荡",
        True,
        True,
        f"BTC 1h 暂无明显单边风险：价格={last.close:g}，MA25={m25:g}，MA99={m99:g}，RSI={rsi:.1f}，ADX={adx:.1f}。",
    )


def classify_weekly_trend(candles: Sequence[Candle], lookback: int = 168) -> WeeklyTrend:
    if len(candles) <= lookback:
        return WeeklyTrend("中性", True, True, "近一周历史K线不足，暂不做周趋势过滤。")

    current = candles[-1]
    start = candles[-lookback - 1]
    week_return = (current.close - start.close) / start.close if start.close > 0 else 0.0
    two_week_return = _pct_change(candles, 336)
    m25 = ma25(candles)
    m99 = ma99(candles)
    slope = _ma25_slope(candles)
    rsi = _rsi(candles, 14)
    adx = _adx(candles, 14)
    if m25 is None or m99 is None or rsi is None or adx is None:
        return WeeklyTrend("中性", True, True, "近一周指标未就绪，暂不做周趋势过滤。")

    long_return_text = "不足" if two_week_return is None else f"{two_week_return * 100:.2f}%"
    bullish = (
        week_return >= 0.025 and current.close > m99 and m25 >= m99 and slope > 0 and rsi >= 50
    ) or (
        two_week_return is not None
        and two_week_return >= 0.04
        and current.close > m99
        and m25 >= m99
        and slope > 0
        and rsi >= 50
    )
    bearish = (
        week_return <= -0.025 and current.close < m99 and m25 <= m99 and slope < 0 and rsi <= 50
    ) or (
        two_week_return is not None
        and two_week_return <= -0.04
        and current.close < m99
        and m25 <= m99
        and slope < 0
        and rsi <= 50
    )
    ma99_distance = abs(current.close - m99) / current.close if current.close > 0 else 0.0
    sideways = (
        abs(week_return) <= 0.01
        and (two_week_return is None or abs(two_week_return) <= 0.02)
        and adx < 15
        and ma99_distance <= 0.015
    )
    if bullish:
        return WeeklyTrend(
            "偏多",
            True,
            False,
            f"近1-2周趋势偏多：7天涨跌幅={week_return * 100:.2f}%，14天涨跌幅={long_return_text}，价格位于MA99上方，过滤空单。",
        )
    if bearish:
        return WeeklyTrend(
            "偏空",
            False,
            True,
            f"近1-2周趋势偏空：7天涨跌幅={week_return * 100:.2f}%，14天涨跌幅={long_return_text}，价格位于MA99下方，过滤多单。",
        )
    if sideways:
        return WeeklyTrend(
            "震荡",
            True,
            True,
            f"近1-2周横盘震荡：7天涨跌幅={week_return * 100:.2f}%，14天涨跌幅={long_return_text}，ADX={adx:.1f}，价格贴近MA99，需要超高评分才允许通过。",
        )
    return WeeklyTrend(
        "中性",
        True,
        True,
        f"近1-2周趋势中性：7天涨跌幅={week_return * 100:.2f}%，14天涨跌幅={long_return_text}，暂不过滤方向。",
    )


def _apply_long_context_filter(
    candidate: ObservationCandidate,
    context: WeeklyTrend,
    *,
    label: str,
    sideways_exception_score: int = SIDEWAYS_EXCEPTION_SCORE,
    sideways_score_penalty: int = SIDEWAYS_SCORE_PENALTY,
) -> ObservationCandidate:
    if candidate.direction not in {"做多观察", "做空观察"}:
        return candidate
    if candidate.direction == "做空观察" and not context.allows_short:
        return replace(
            candidate,
            score=min(candidate.score, 59),
            level="暂不关注",
            reasons=candidate.reasons + (f"{label}过滤：{context.detail}",),
        )
    if candidate.direction == "做多观察" and not context.allows_long:
        return replace(
            candidate,
            score=min(candidate.score, 59),
            level="暂不关注",
            reasons=candidate.reasons + (f"{label}过滤：{context.detail}",),
        )
    if context.direction == "震荡":
        if candidate.score < sideways_exception_score:
            return replace(
                candidate,
                score=min(candidate.score, 69),
                level="暂不关注",
                reasons=candidate.reasons
                + (
                    f"{label}横盘过滤：{context.detail} 当前评分 {candidate.score}，未达到超高分例外阈值 {sideways_exception_score}。",
                ),
            )
        new_score = _clamp_score(candidate.score - sideways_score_penalty)
        return replace(
            candidate,
            score=new_score,
            level=_level(new_score),
            reasons=candidate.reasons
            + (
                f"{label}横盘降权：{context.detail} 评分达到超高分例外阈值，扣{sideways_score_penalty}分后继续观察。",
            ),
        )
    return replace(candidate, reasons=candidate.reasons + (f"{label}通过：{context.detail}",))


def _build_candidate_prices(
    *,
    direction: str,
    current: float,
    trigger: float,
    recent: Sequence[Candle],
    atr: float,
    target_rr: float,
) -> tuple[float, float, float]:
    if direction == "做多观察":
        structure_stop = min(c.low for c in recent) - 0.25 * atr
        atr_stop = current - 1.3 * atr
        stop = max(structure_stop, atr_stop)
        risk = max(trigger - stop, trigger * 0.001)
        return stop, trigger + target_rr * risk, trigger + (target_rr + 1.0) * risk

    structure_stop = max(c.high for c in recent) + 0.25 * atr
    atr_stop = current + 1.3 * atr
    stop = min(structure_stop, atr_stop)
    risk = max(stop - trigger, trigger * 0.001)
    return stop, trigger - target_rr * risk, trigger - (target_rr + 1.0) * risk


def _empty_candidate(
    *,
    symbol: str,
    direction: str,
    price: float,
    expected_hold_hours: float,
    target_rr: float,
    expires_after_minutes: int,
    reason: str,
) -> ObservationCandidate:
    return ObservationCandidate(
        symbol=symbol.strip().upper(),
        direction=direction,
        score=0,
        level="暂不关注",
        current_price=price,
        trigger_price=price,
        stop_loss=price,
        take_profit_1=price,
        take_profit_2=price,
        target_rr=target_rr,
        expires_after_minutes=expires_after_minutes,
        expected_hold_hours=expected_hold_hours,
        reasons=(reason,),
    )


def _legacy_score_candidate_direction(
    *,
    direction: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    expected_hold_hours: float,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    symbol: str,
    target_rr: float,
    expires_after_minutes: int,
) -> ObservationCandidate:
    is_long = direction == "做多观察"
    trend_last = trend_candles[-1]
    entry_last = entry_candles[-1]
    trend_ma25 = ma25(trend_candles)
    trend_ma99 = ma99(trend_candles)
    entry_ma25 = ma25(entry_candles)
    entry_ma99 = ma99(entry_candles)
    vma = volume_ma(entry_candles, 20)
    atr = _avg_true_range(entry_candles, 14)
    rsi = _rsi(entry_candles, 14)
    adx = _adx(entry_candles, 14)

    if None in {trend_ma25, trend_ma99, entry_ma25, entry_ma99, vma, atr, rsi, adx}:
        return _empty_candidate(
            symbol=symbol,
            direction=direction,
            price=entry_last.close,
            expected_hold_hours=expected_hold_hours,
            target_rr=target_rr,
            expires_after_minutes=expires_after_minutes,
            reason="指标尚未就绪。",
        )

    assert trend_ma25 is not None
    assert trend_ma99 is not None
    assert entry_ma25 is not None
    assert entry_ma99 is not None
    assert vma is not None
    assert atr is not None
    assert rsi is not None
    assert adx is not None

    score = 0.0
    trend_pts = 0.0
    momentum_pts = 0.0
    volume_pts = 0.0
    risk_pts = 0.0
    reasons: list[str] = []

    slope = _ma25_slope(trend_candles)
    ma_alignment = trend_ma25 > trend_ma99 if is_long else trend_ma25 < trend_ma99
    trend_ok = trend_last.close > trend_ma99 if is_long else trend_last.close < trend_ma99
    ma25_ok = trend_last.close > trend_ma25 if is_long else trend_last.close < trend_ma25
    slope_ok = slope > 0 if is_long else slope < 0

    if ma_alignment:
        score += 12
        trend_pts += 12
        reasons.append("1h MA25 与 MA99 排列支持当前方向。")
    if trend_ok:
        score += 12
        trend_pts += 12
        reasons.append("1h 价格位于 MA99 的有利方向，过滤掉一部分逆势交易。")
    if ma25_ok and slope_ok:
        score += 10
        trend_pts += 10
        reasons.append("1h 短中期方向和均线斜率一致。")

    # Add short-term (6h) momentum and 1h trend strength to reduce noisy flips.
    trend_rsi = _rsi(trend_candles, 14)
    trend_adx = _adx(trend_candles, 14)
    move_6h = _directional_pct_change(trend_candles, 6, direction)
    if move_6h is not None:
        if move_6h >= 0.003:
            score += 8
            momentum_pts += 8
            reasons.append(f"1h 近6小时顺势动量较强（{move_6h * 100:.2f}%）。")
        elif move_6h <= -0.001:
            score -= 8
            momentum_pts -= 8
            reasons.append(f"1h 近6小时动量与方向相反（{move_6h * 100:.2f}%），本轮信号降噪扣分。")

    if trend_rsi is not None:
        if is_long and trend_rsi >= 52:
            score += 4
            trend_pts += 4
            reasons.append(f"1h RSI={trend_rsi:.1f} 偏强，偏向顺势做多。")
        elif (not is_long) and trend_rsi <= 48:
            score += 4
            trend_pts += 4
            reasons.append(f"1h RSI={trend_rsi:.1f} 偏弱，偏向顺势做空。")
        elif is_long and trend_rsi <= 46:
            score -= 4
            trend_pts -= 4
            reasons.append(f"1h RSI={trend_rsi:.1f} 偏弱，做多延续性存疑，扣分。")
        elif (not is_long) and trend_rsi >= 54:
            score -= 4
            trend_pts -= 4
            reasons.append(f"1h RSI={trend_rsi:.1f} 偏强，做空延续性存疑，扣分。")

    if trend_adx is not None:
        if trend_adx >= 18:
            score += 4
            trend_pts += 4
            reasons.append(f"1h ADX={trend_adx:.1f} 显示有趋势强度。")
        elif trend_adx < 12:
            score -= 4
            trend_pts -= 4
            reasons.append(f"1h ADX={trend_adx:.1f} 偏低，震荡概率高，扣分。")

    eff = _efficiency_ratio(trend_candles, TREND_EFFICIENCY_LOOKBACK_BARS)
    if eff is not None:
        eff_abs = abs(eff)
        if eff_abs < MIN_TREND_EFFICIENCY_ABS:
            score -= 6
            trend_pts -= 6
            reasons.append(
                f"1h 趋势效率偏低（ER={eff:+.3f}），偏震荡，容易出现多空来回，降噪扣6分。"
            )
        else:
            if is_long and eff > 0:
                score += 4
                trend_pts += 4
                reasons.append(f"1h 趋势效率支持做多（ER={eff:+.3f}）。")
            elif (not is_long) and eff < 0:
                score += 4
                trend_pts += 4
                reasons.append(f"1h 趋势效率支持做空（ER={eff:+.3f}）。")
            else:
                score -= 6
                trend_pts -= 6
                reasons.append(f"1h 趋势效率与方向相反（ER={eff:+.3f}），降噪扣6分。")

    recent = entry_candles[-(ENTRY_RANGE_LOOKBACK_BARS + 1) : -1]
    recent_range_high = max(c.high for c in recent)
    recent_range_low = min(c.low for c in recent)
    range_pct = (recent_range_high - recent_range_low) / entry_last.close

    if 0.003 <= range_pct <= 0.035:
        score += 8
        risk_pts += 8
        reasons.append("15m 近期波动区间适中，止损和目标更容易设计。")

    if adx >= 18:
        score += 8
        momentum_pts += 8
        reasons.append("15m ADX 显示有一定趋势强度，减少纯震荡假突破。")
    elif adx < 12:
        score -= 6
        momentum_pts -= 6
        reasons.append("15m ADX 偏低，市场偏震荡，候选质量降低（扣6分）。")

    if is_long:
        pulled_back = recent_range_low <= entry_ma25 * 1.004 or recent_range_low <= entry_ma99 * 1.01
        not_overheated = 42 <= rsi <= 68
        trigger = max(entry_last.high, recent_range_high) + 0.08 * atr
        trigger_gap = (trigger - entry_last.close) / entry_last.close
        above_range_mid = entry_last.close >= (recent_range_high + recent_range_low) / 2
        if pulled_back:
            score += 20
            momentum_pts += 20
            reasons.append("15m 有回踩均线或结构区的动作，不是单纯追高。")
        if not_overheated:
            score += 10
            momentum_pts += 10
            reasons.append("15m RSI 处于相对健康区间，暂未明显过热。")
        if entry_last.close >= entry_ma25:
            score += 8
            momentum_pts += 8
            reasons.append("15m 当前价格已经重新靠近或站上 MA25。")
        if above_range_mid:
            score += 8
            momentum_pts += 8
            reasons.append("15m 价格回到近期结构中轴上方，突破触发更有意义。")
    else:
        pulled_back = recent_range_high >= entry_ma25 * 0.996 or recent_range_high >= entry_ma99 * 0.99
        not_overheated = 32 <= rsi <= 58
        trigger = min(entry_last.low, recent_range_low) - 0.08 * atr
        trigger_gap = (entry_last.close - trigger) / entry_last.close
        below_range_mid = entry_last.close <= (recent_range_high + recent_range_low) / 2
        if pulled_back:
            score += 20
            momentum_pts += 20
            reasons.append("15m 有反弹到均线或结构区的动作，不是单纯追空。")
        if not_overheated:
            score += 10
            momentum_pts += 10
            reasons.append("15m RSI 处于相对健康区间，暂未明显超跌。")
        if entry_last.close <= entry_ma25:
            score += 8
            momentum_pts += 8
            reasons.append("15m 当前价格已经重新靠近或跌回 MA25 下方。")
        if below_range_mid:
            score += 8
            momentum_pts += 8
            reasons.append("15m 价格回到近期结构中轴下方，跌破触发更有意义。")

    if 0.0004 <= trigger_gap <= 0.006:
        score += 10
        momentum_pts += 10
        reasons.append("触发价距离当前价格不远，适合短时间确认。")
    elif trigger_gap <= 0.012:
        score += 5
        momentum_pts += 5
        reasons.append("触发价略远，需要等待更明确的突破确认。")
    else:
        score -= 6
        momentum_pts -= 6
        reasons.append("触发价距离偏远，本轮候选质量降低（扣6分）。")

    if entry_last.volume >= 1.25 * vma:
        score += 12
        volume_pts += 12
        reasons.append("15m 成交量明显高于近20根均量，参与度较高。")
    elif entry_last.volume >= 0.9 * vma:
        score += 8
        volume_pts += 8
        reasons.append("15m 成交量不弱，具备基本确认。")

    stop, tp1, tp2 = _build_candidate_prices(
        direction=direction,
        current=entry_last.close,
        trigger=trigger,
        recent=entry_candles[-ENTRY_STOP_LOOKBACK_BARS:],
        atr=atr,
        target_rr=target_rr,
    )
    risk_pct = abs(trigger - stop) / trigger
    if 0.0025 <= risk_pct <= 0.014:
        score += 10
        risk_pts += 10
        reasons.append("止损距离相对合理，能支持更高盈亏比。")
    elif risk_pct <= 0.022:
        score += 5
        risk_pts += 5
        reasons.append("止损略宽，若触发也应降低仓位。")
    else:
        reasons.append("止损距离偏宽，候选质量降低。")

    position = _position_context(
        direction=direction,
        trend_candles=trend_candles,
        trigger_price=trigger,
        stop_loss=stop,
    )
    score += position.score_adjustment
    risk_pts += position.score_adjustment
    if position.score_adjustment > 0:
        reasons.append(f"位置过滤加分：{position.detail}")
    elif position.score_adjustment < 0:
        reasons.append(f"位置过滤扣分：{position.detail}")
    else:
        reasons.append(f"位置过滤中性：{position.detail}")

    if not reasons:
        reasons.append("当前结构不清晰，暂不适合重点观察。")

    hold_plan = estimate_hold_time_plan(
        direction=direction,
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        base_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
    )
    # Use hold-time plan as a proxy for "trend stage/strength" quality.
    # Bias toward early/mid-stage trends and penalize late/overheated stages,
    # since those tend to reduce near-term directional follow-through.
    hold_adjust = 0
    if hold_plan.stage == "趋势初段":
        hold_adjust += 4
    elif hold_plan.stage == "趋势中段":
        hold_adjust += 2
    elif hold_plan.stage == "趋势末段/过热":
        hold_adjust -= 6
    elif hold_plan.stage == "趋势未确认":
        hold_adjust -= 3
    if hold_plan.strength == "强":
        hold_adjust += 2
    elif hold_plan.strength == "弱":
        hold_adjust -= 2
    if hold_adjust:
        score += hold_adjust
        trend_pts += hold_adjust
        if hold_adjust > 0:
            reasons.append(f"持仓因子加分：趋势阶段={hold_plan.stage}，强度={hold_plan.strength}，加{hold_adjust}分。")
        else:
            reasons.append(f"持仓因子扣分：趋势阶段={hold_plan.stage}，强度={hold_plan.strength}，扣{abs(hold_adjust)}分。")
    reasons.append(f"持仓时间因子：{hold_plan.detail}")
    final_score = min(99, _clamp_score(score))
    reasons.append(
        f"评分拆分：趋势={trend_pts:.0f} 动量={momentum_pts:.0f} 成交量={volume_pts:.0f} 风险={risk_pts:.0f} 总分={final_score}/99"
    )

    return ObservationCandidate(
        symbol=symbol.strip().upper(),
        direction=direction,
        score=final_score,
        level=_level(final_score),
        current_price=entry_last.close,
        trigger_price=trigger,
        stop_loss=stop,
        take_profit_1=tp1,
        take_profit_2=tp2,
        target_rr=target_rr,
        expires_after_minutes=expires_after_minutes,
        expected_hold_hours=hold_plan.hold_hours,
        reasons=tuple(reasons),
    )


def _soft_score_and_hard_filters(
    *,
    legacy: ObservationCandidate,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    expected_hold_hours: float,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    weekly_filter: bool,
    hard_volume_filter: bool,
    hard_short_trend_filter: bool,
    market_regime: MarketRegimeResult | None = None,
    market_candles: Sequence[Candle] | None = None,
    market_entry_candles: Sequence[Candle] | None = None,
) -> ObservationCandidate:
    """
    Convert the legacy candidate into:
      - hard filters (gate paper-trade entry)
      - soft score (ranking only)
    """
    direction = legacy.direction
    is_long = direction == "做多观察"
    symbol = legacy.symbol.strip().upper()
    entry_last = entry_candles[-1]

    # Indicators (optional/defensive: do NOT fabricate if unavailable).
    trend_ma25 = ma25(trend_candles)
    trend_ma99 = ma99(trend_candles)
    entry_ma25 = ma25(entry_candles)
    entry_ma99 = ma99(entry_candles)
    vma = volume_ma(entry_candles, 20)
    atr = _avg_true_range(entry_candles, 14)
    rsi = _rsi(entry_candles, 14)
    adx = _adx(entry_candles, 14)
    trend_rsi = _rsi(trend_candles, 14)
    er = _efficiency_ratio(trend_candles, TREND_EFFICIENCY_LOOKBACK_BARS)
    move_6h = _directional_pct_change(trend_candles, 6, direction)

    regime = market_regime or detect_market_regime(trend_candles=trend_candles, entry_candles=entry_candles)

    hard_status: dict[str, str] = {}
    hard_detail: dict[str, str] = {}
    failed: list[str] = []
    unavailable: list[str] = []

    def _append_detail(name: str, detail: str) -> None:
        if not detail:
            return
        prev = hard_detail.get(name, "")
        hard_detail[name] = f"{prev}；{detail}" if prev else detail

    def _pass(name: str, detail: str) -> None:
        if hard_status.get(name) == "fail":
            return
        hard_status[name] = "pass"
        _append_detail(name, detail)

    def _fail(name: str, detail: str) -> None:
        hard_status[name] = "fail"
        _append_detail(name, detail)
        failed.append(name)

    def _unavailable(name: str, detail: str) -> None:
        if name in hard_status:
            return
        hard_status[name] = "unavailable"
        hard_detail[name] = detail
        unavailable.append(name)

    # --- hard filters ---
    # Market regime filter (stateful filter beyond pure indicator thresholds)
    if regime.regime == "unknown":
        _fail("not_in_chop_zone", "市场状态=unknown（数据不足/状态不明），保守起见不允许交易。")
    if regime.no_trade_zone:
        _fail("not_in_chop_zone", "no_trade_zone=true（价格在区间中部40%-60%），不允许交易。")
    if regime.regime == "range":
        _fail("not_in_chop_zone", "市场状态=range（震荡市），不允许 breakout 直接开单。")
    # compression: allowed, but must wait for trigger confirmation (paper default confirm mode).

    # 1) higher_timeframe_trend_aligned
    if trend_ma25 is None or trend_ma99 is None:
        _unavailable("higher_timeframe_trend_aligned", "1h MA25/MA99 未就绪")
    else:
        slope = _ma25_slope(trend_candles)
        ma_alignment = trend_ma25 >= trend_ma99 if is_long else trend_ma25 <= trend_ma99
        price_ok = trend_candles[-1].close >= trend_ma99 if is_long else trend_candles[-1].close <= trend_ma99
        slope_ok = slope >= 0 if is_long else slope <= 0
        ok = ma_alignment and price_ok and slope_ok
        _pass(
            "higher_timeframe_trend_aligned",
            f"1h结构：MA25={trend_ma25:.4g} MA99={trend_ma99:.4g} slope={slope:.4g} close={trend_candles[-1].close:g}",
        ) if ok else _fail(
            "higher_timeframe_trend_aligned",
            f"1h结构不一致：MA25={trend_ma25:.4g} MA99={trend_ma99:.4g} slope={slope:.4g} close={trend_candles[-1].close:g}",
        )

    if weekly_filter:
        weekly = classify_weekly_trend(trend_candles)
        if is_long and not weekly.allows_long:
            _fail("higher_timeframe_trend_aligned", f"单币周趋势过滤：{weekly.detail}")
        elif (not is_long) and not weekly.allows_short:
            _fail("higher_timeframe_trend_aligned", f"单币周趋势过滤：{weekly.detail}")

    # 15m结构（触发价附近）+（非BTC时）BTC同向结构
    if hard_short_trend_filter:
        ok_ema, ema_detail = _short_term_trend_ok(entry_candles, direction, reference_price=legacy.trigger_price)
        if not ok_ema:
            _fail("higher_timeframe_trend_aligned", f"15m EMA结构不一致：{ema_detail}")

        if symbol != "BTCUSDT":
            if market_entry_candles is None:
                _unavailable("higher_timeframe_trend_aligned", "BTC 15m K线缺失，无法校验BTC同向结构")
            else:
                btc_ok, btc_detail = _short_term_trend_ok(market_entry_candles, direction)
                if not btc_ok:
                    _fail("higher_timeframe_trend_aligned", f"BTC 15m同向结构不一致：{btc_detail}")
    else:
        _unavailable("higher_timeframe_trend_aligned", "短周期EMA结构过滤已关闭")

    # BTC整体周趋势/环境过滤（非BTC时）
    if symbol != "BTCUSDT" and market_candles is not None:
        market_weekly = classify_weekly_trend(market_candles)
        if is_long and not market_weekly.allows_long:
            _fail("higher_timeframe_trend_aligned", f"BTC周趋势过滤：{market_weekly.detail}")
        elif (not is_long) and not market_weekly.allows_short:
            _fail("higher_timeframe_trend_aligned", f"BTC周趋势过滤：{market_weekly.detail}")
        regime = classify_market_regime(market_candles)
        if is_long and not regime.allows_long:
            _fail("higher_timeframe_trend_aligned", f"大盘过滤：{regime.detail}")
        elif (not is_long) and not regime.allows_short:
            _fail("higher_timeframe_trend_aligned", f"大盘过滤：{regime.detail}")

    # 2) not_in_chop_zone
    # Note: we intentionally keep this independent of score; if some data is missing, mark unavailable.
    if adx is None:
        _unavailable("not_in_chop_zone", "15m ADX 未就绪")
    elif adx < 12:
        _fail("not_in_chop_zone", f"15m ADX过低：ADX={adx:.1f}（<12，偏震荡）")
    else:
        _pass("not_in_chop_zone", f"15m ADX={adx:.1f}")
    if er is None:
        _unavailable("not_in_chop_zone", "1h ER 未就绪")
    elif abs(er) < MIN_TREND_EFFICIENCY_ABS:
        _fail("not_in_chop_zone", f"1h ER过低：ER={er:.3f}（|ER|<{MIN_TREND_EFFICIENCY_ABS}，偏无趋势）")
    else:
        # Only mark pass if not already failed by previous checks.
        if hard_status.get("not_in_chop_zone") != "fail":
            _pass("not_in_chop_zone", f"1h ER={er:.3f}")

    if hard_volume_filter:
        vol_ok, vol_detail = _volume_spike_ok(entry_candles)
        if not vol_ok:
            _fail("not_in_chop_zone", f"成交量过滤：{vol_detail}")
        else:
            if hard_status.get("not_in_chop_zone") != "fail":
                _pass("not_in_chop_zone", f"成交量通过：{vol_detail}")
    else:
        _unavailable("not_in_chop_zone", "成交量过滤已关闭")

    # 3) trigger_price_valid
    if legacy.trigger_price <= 0 or legacy.current_price <= 0:
        _fail("trigger_price_valid", "价格非法")
    else:
        if is_long and legacy.trigger_price <= legacy.current_price:
            _fail("trigger_price_valid", f"做多触发价需高于现价：trigger={legacy.trigger_price:g} <= now={legacy.current_price:g}")
        elif (not is_long) and legacy.trigger_price >= legacy.current_price:
            _fail("trigger_price_valid", f"做空触发价需低于现价：trigger={legacy.trigger_price:g} >= now={legacy.current_price:g}")
        else:
            gap = abs(legacy.trigger_price - legacy.current_price) / legacy.current_price
            if gap < 0.0004:
                _fail("trigger_price_valid", f"触发价过近：gap={gap*100:.3f}%（<0.04%）")
            elif gap > 0.012:
                _fail("trigger_price_valid", f"触发价过远：gap={gap*100:.3f}%（>1.2%）")
            else:
                _pass("trigger_price_valid", f"gap={gap*100:.3f}%")

    # 4) stop_distance_reasonable
    if legacy.trigger_price <= 0:
        _fail("stop_distance_reasonable", "触发价非法")
    else:
        risk_pct = abs(legacy.trigger_price - legacy.stop_loss) / legacy.trigger_price
        if is_long and legacy.stop_loss >= legacy.trigger_price:
            _fail("stop_distance_reasonable", f"做多止损需低于触发价：stop={legacy.stop_loss:g} trigger={legacy.trigger_price:g}")
        elif (not is_long) and legacy.stop_loss <= legacy.trigger_price:
            _fail("stop_distance_reasonable", f"做空止损需高于触发价：stop={legacy.stop_loss:g} trigger={legacy.trigger_price:g}")
        elif risk_pct < 0.0025:
            _fail("stop_distance_reasonable", f"止损过紧：risk={risk_pct*100:.3f}%（<0.25%）")
        elif risk_pct > 0.022:
            _fail("stop_distance_reasonable", f"止损过宽：risk={risk_pct*100:.3f}%（>2.2%）")
        else:
            _pass("stop_distance_reasonable", f"risk={risk_pct*100:.3f}%")

    # 5) tp1_reachable_by_atr
    if atr is None or atr <= 0:
        _unavailable("tp1_reachable_by_atr", "ATR 未就绪")
    else:
        risk = abs(legacy.trigger_price - legacy.stop_loss)
        if risk <= 0:
            _fail("tp1_reachable_by_atr", "risk<=0")
        else:
            if expected_hold_hours <= 1.5:
                max_ratio = 1.8
            elif expected_hold_hours <= 2.5:
                max_ratio = 2.2
            else:
                max_ratio = 2.6
            ratio = risk / atr
            if ratio > max_ratio:
                _fail("tp1_reachable_by_atr", f"risk/ATR={ratio:.2f}（>{max_ratio:.2f}，TP1=1R可能偏远）")
            else:
                _pass("tp1_reachable_by_atr", f"risk/ATR={ratio:.2f}（<= {max_ratio:.2f}）")

    # 6) spread_or_liquidity_ok (optional)
    _unavailable("spread_or_liquidity_ok", "暂无盘口/点差数据源，未启用该过滤")

    # --- soft score (ranking only) ---
    # Soft score must NOT bypass hard filters; it only ranks candidates that already pass.
    trend_score = 0.0
    momentum_score = 0.0
    volume_score = 0.0
    risk_score = 0.0
    btc_env_score = 0.0
    hold_score = 0.0

    if trend_ma25 is not None and trend_ma99 is not None:
        ma_alignment = trend_ma25 >= trend_ma99 if is_long else trend_ma25 <= trend_ma99
        trend_score += 10.0 if ma_alignment else 0.0
        price_ok = trend_candles[-1].close >= trend_ma99 if is_long else trend_candles[-1].close <= trend_ma99
        trend_score += 8.0 if price_ok else 0.0
        slope = _ma25_slope(trend_candles)
        slope_ok = slope >= 0 if is_long else slope <= 0
        trend_score += 6.0 if slope_ok else 0.0
    if trend_rsi is not None:
        if is_long and trend_rsi >= 52:
            trend_score += 3.0
        elif (not is_long) and trend_rsi <= 48:
            trend_score += 3.0

    # Momentum (15m)
    if move_6h is not None:
        if move_6h >= 0.003:
            momentum_score += 6.0
        elif move_6h >= 0.001:
            momentum_score += 3.0
        elif move_6h <= -0.001:
            momentum_score -= 3.0
    if adx is not None:
        if adx >= 22:
            momentum_score += 6.0
        elif adx >= 18:
            momentum_score += 5.0
        elif adx >= 14:
            momentum_score += 3.0
        elif adx >= 12:
            momentum_score += 1.0
        else:
            momentum_score -= 2.0
    if rsi is not None:
        if is_long:
            if 45 <= rsi <= 65:
                momentum_score += 5.0
            elif 65 < rsi <= 72:
                momentum_score += 2.0
            elif rsi < 40 or rsi > 75:
                momentum_score -= 1.5
        else:
            if 35 <= rsi <= 55:
                momentum_score += 5.0
            elif 28 <= rsi < 35:
                momentum_score += 2.0
            elif rsi < 25 or rsi > 60:
                momentum_score -= 1.5

    if entry_ma25 is not None and entry_ma99 is not None:
        recent = entry_candles[-ENTRY_RANGE_LOOKBACK_BARS:]
        recent_high = max(c.high for c in recent)
        recent_low = min(c.low for c in recent)
        pulled_back = (
            (recent_low <= entry_ma25 * 1.004 or recent_low <= entry_ma99 * 1.01)
            if is_long
            else (recent_high >= entry_ma25 * 0.996 or recent_high >= entry_ma99 * 0.99)
        )
        momentum_score += 5.0 if pulled_back else 0.0

        # Trigger gap as quality proxy.
        if legacy.current_price > 0:
            trigger_gap = abs(legacy.trigger_price - legacy.current_price) / legacy.current_price
            if trigger_gap <= 0.006:
                momentum_score += 2.0
            elif trigger_gap <= 0.012:
                momentum_score += 1.0

    # Volume
    if vma is not None and vma > 0:
        multiple = entry_last.volume / vma
        if multiple >= 1.6:
            volume_score += 15.0
        elif multiple >= 1.3:
            volume_score += 12.0
        elif multiple >= 1.1:
            volume_score += 8.0
        elif multiple >= 0.9:
            volume_score += 5.0

    # Risk (stop distance + position)
    if legacy.trigger_price > 0:
        risk_pct = abs(legacy.trigger_price - legacy.stop_loss) / legacy.trigger_price
        if 0.0025 <= risk_pct <= 0.012:
            risk_score += 12.0
        elif risk_pct <= 0.018:
            risk_score += 8.0
        elif risk_pct <= 0.022:
            risk_score += 5.0
        else:
            risk_score += 0.0
    pos = _position_context(
        direction=direction,
        trend_candles=trend_candles,
        trigger_price=legacy.trigger_price,
        stop_loss=legacy.stop_loss,
    )
    if pos.score_adjustment >= 0:
        risk_score += 4.0

    # BTC environment (optional)
    if symbol != "BTCUSDT" and market_candles is not None:
        btc_env_score += 2.0
        sym_ret = _pct_change(trend_candles, 6)
        btc_ret = _pct_change(market_candles, 6)
        if sym_ret is not None and btc_ret is not None:
            rel = sym_ret - btc_ret
            if is_long and rel >= 0.002:
                btc_env_score += 4.0
            elif (not is_long) and rel <= -0.002:
                btc_env_score += 4.0
        regime = classify_market_regime(market_candles)
        if (is_long and regime.direction == "偏强") or ((not is_long) and regime.direction == "偏弱"):
            btc_env_score += 4.0

    # Hold-time plan as "stage/strength" quality
    hold_plan = estimate_hold_time_plan(
        direction=direction,
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        base_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
    )
    if hold_plan.stage == "趋势初段":
        hold_score += 3.0
    elif hold_plan.stage == "趋势中段":
        hold_score += 1.5
    elif hold_plan.stage == "趋势末段/过热":
        hold_score -= 4.0
    elif hold_plan.stage == "趋势未确认":
        hold_score -= 2.0
    if hold_plan.strength == "强":
        hold_score += 2.0
    elif hold_plan.strength == "弱":
        hold_score -= 1.0

    # Clamp component ranges to keep score distribution sane.
    trend_score = max(0.0, min(30.0, trend_score))
    momentum_score = max(0.0, min(25.0, momentum_score))
    volume_score = max(0.0, min(15.0, volume_score))
    risk_score = max(0.0, min(20.0, risk_score))
    btc_env_score = max(0.0, min(10.0, btc_env_score))
    hold_score = max(-5.0, min(5.0, hold_score))

    raw_score = trend_score + momentum_score + volume_score + risk_score + btc_env_score + hold_score
    # Light up the midrange a bit to keep practical thresholds (e.g. 70/80/90) meaningful,
    # without letting everything saturate at 99/100 again.
    final_score = _clamp_score(raw_score * 1.10)

    # Deduplicate failed filter keys while keeping insertion order.
    if failed:
        seen: set[str] = set()
        failed = [name for name in failed if not (name in seen or seen.add(name))]
    hard_passed = len(failed) == 0
    # If some filters are unavailable, they do NOT block entry yet; we only log them.
    components = {
        "market_regime": {
            "regime": regime.regime,
            "confidence": int(regime.confidence),
            "no_trade_zone": bool(regime.no_trade_zone),
            "reasons": list(regime.reasons),
        },
        "soft_score": {
            "trend": round(trend_score, 6),
            "momentum": round(momentum_score, 6),
            "volume": round(volume_score, 6),
            "risk": round(risk_score, 6),
            "btc_env": round(btc_env_score, 6),
            "hold_time": round(hold_score, 6),
        },
        "hard_filters": {
            name: {
                "status": hard_status.get(name, "unavailable"),
                "detail": hard_detail.get(name, ""),
            }
            for name in (
                "higher_timeframe_trend_aligned",
                "not_in_chop_zone",
                "trigger_price_valid",
                "stop_distance_reasonable",
                "tp1_reachable_by_atr",
                "spread_or_liquidity_ok",
            )
        },
        "unavailable_hard_filters": unavailable,
        "indicators": {
            "adx_15m": round(float(adx), 6) if adx is not None else None,
            "atr_15m": round(float(atr), 8) if atr is not None else None,
            "er_1h": round(float(er), 6) if er is not None else None,
            "rsi_15m": round(float(rsi), 6) if rsi is not None else None,
            "vma_20": round(float(vma), 6) if vma is not None else None,
        },
    }
    components_json = json.dumps(components, ensure_ascii=False, separators=(",", ":"))

    legacy_reasons: list[str] = []
    for r in legacy.reasons:
        if r.startswith("评分拆分："):
            legacy_reasons.append(r.replace("评分拆分：", "legacy评分拆分：", 1))
        else:
            legacy_reasons.append(r)
    regime_line = (
        f"市场状态：{regime.regime}（confidence={int(regime.confidence)}/100，no_trade_zone={bool(regime.no_trade_zone)}）"
    )
    hard_line = (
        f"硬过滤：{'通过' if hard_passed else '失败'}；failed={','.join(failed) if failed else '-'}；"
        f"unavailable={','.join(unavailable) if unavailable else '-'}"
    )
    soft_line = (
        "soft评分拆分："
        f"trend={trend_score:.0f} momentum={momentum_score:.0f} volume={volume_score:.0f} risk={risk_score:.0f} "
        f"btc={btc_env_score:.0f} hold={hold_score:.0f} raw={raw_score:.1f} -> final={final_score}/100"
    )

    return replace(
        legacy,
        score=final_score,
        level=_level(final_score),
        legacy_score=int(legacy.score),
        hard_filter_passed=hard_passed,
        failed_hard_filters=tuple(failed),
        raw_score=float(raw_score),
        final_score=int(final_score),
        score_components_json=components_json,
        market_regime=str(regime.regime),
        market_regime_confidence=int(regime.confidence),
        no_trade_zone=bool(regime.no_trade_zone),
        market_regime_reasons=tuple(regime.reasons),
        reasons=tuple(legacy_reasons + [regime_line, hard_line, soft_line]),
        # Keep hold plan hours from legacy candidate (already dynamic); don't overwrite.
    )


def _score_candidate_direction(
    *,
    direction: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    expected_hold_hours: float,
    min_hold_hours: float,
    max_hold_hours: float,
    dynamic_hold: bool,
    symbol: str,
    target_rr: float,
    expires_after_minutes: int,
    weekly_filter: bool = True,
    hard_volume_filter: bool = True,
    hard_short_trend_filter: bool = True,
    market_regime: MarketRegimeResult | None = None,
    market_candles: Sequence[Candle] | None = None,
    market_entry_candles: Sequence[Candle] | None = None,
) -> ObservationCandidate:
    legacy = _legacy_score_candidate_direction(
        direction=direction,
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        expected_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
        symbol=symbol,
        target_rr=target_rr,
        expires_after_minutes=expires_after_minutes,
    )
    return _soft_score_and_hard_filters(
        legacy=legacy,
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        expected_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
        weekly_filter=weekly_filter,
        hard_volume_filter=hard_volume_filter,
        hard_short_trend_filter=hard_short_trend_filter,
        market_regime=market_regime,
        market_candles=market_candles,
        market_entry_candles=market_entry_candles,
    )


def evaluate_observation_candidate(
    *,
    symbol: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    expected_hold_hours: float = 2.0,
    min_hold_hours: float = 1.0,
    max_hold_hours: float = 4.0,
    dynamic_hold: bool = True,
    target_rr: float = 2.0,
    expires_after_minutes: int = 20,
    market_candles: Sequence[Candle] | None = None,
    market_entry_candles: Sequence[Candle] | None = None,
    weekly_filter: bool = True,
    hard_volume_filter: bool = True,
    hard_short_trend_filter: bool = True,
) -> ObservationCandidate:
    if len(trend_candles) < MIN_TREND_BARS or len(entry_candles) < MIN_ENTRY_BARS:
        price = entry_candles[-1].close if entry_candles else 0.0
        return _empty_candidate(
            symbol=symbol,
            direction="暂无信号",
            price=price,
            expected_hold_hours=expected_hold_hours,
            target_rr=target_rr,
            expires_after_minutes=expires_after_minutes,
            reason="历史K线不足，暂时无法评价。",
        )

    symbol = symbol.strip().upper()
    regime = detect_market_regime(trend_candles=trend_candles, entry_candles=entry_candles)

    def _mark_chop(candidate: ObservationCandidate, gap: int) -> ObservationCandidate:
        # Hard filter: not_in_chop_zone. Do not let score override this.
        failed = list(candidate.failed_hard_filters)
        if "not_in_chop_zone" not in failed:
            failed.append("not_in_chop_zone")
        components_json = candidate.score_components_json
        try:
            payload = json.loads(components_json) if components_json else {}
            hard = payload.get("hard_filters", {})
            item = hard.get("not_in_chop_zone", {"status": "unavailable", "detail": ""})
            item["status"] = "fail"
            extra = f"方向置信度过低：多空评分差距仅 {gap} 分（阈值 {MIN_DIRECTION_SCORE_GAP}）"
            if item.get("detail"):
                item["detail"] = f"{item['detail']}；{extra}"
            else:
                item["detail"] = extra
            hard["not_in_chop_zone"] = item
            payload["hard_filters"] = hard
            components_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # Defensive: do not fail the strategy if JSON formatting breaks.
            components_json = candidate.score_components_json
        return replace(
            candidate,
            hard_filter_passed=False,
            failed_hard_filters=tuple(failed),
            score_components_json=components_json,
        )

    long_candidate = _score_candidate_direction(
        direction="做多观察",
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        expected_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
        symbol=symbol,
        target_rr=target_rr,
        expires_after_minutes=expires_after_minutes,
        weekly_filter=weekly_filter,
        hard_volume_filter=hard_volume_filter,
        hard_short_trend_filter=hard_short_trend_filter,
        market_regime=regime,
        market_candles=None if symbol == "BTCUSDT" else market_candles,
        market_entry_candles=None if symbol == "BTCUSDT" else market_entry_candles,
    )
    short_candidate = _score_candidate_direction(
        direction="做空观察",
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        expected_hold_hours=expected_hold_hours,
        min_hold_hours=min_hold_hours,
        max_hold_hours=max_hold_hours,
        dynamic_hold=dynamic_hold,
        symbol=symbol,
        target_rr=target_rr,
        expires_after_minutes=expires_after_minutes,
        weekly_filter=weekly_filter,
        hard_volume_filter=hard_volume_filter,
        hard_short_trend_filter=hard_short_trend_filter,
        market_regime=regime,
        market_candles=None if symbol == "BTCUSDT" else market_candles,
        market_entry_candles=None if symbol == "BTCUSDT" else market_entry_candles,
    )

    # If both directions score similarly, it's usually a noisy / mean-reverting state.
    # Skip these to improve directional accuracy and avoid alert spam in chop.
    if long_candidate.direction in {"做多观察", "做空观察"} and short_candidate.direction in {"做多观察", "做空观察"}:
        gap = abs(long_candidate.score - short_candidate.score)
        if gap < MIN_DIRECTION_SCORE_GAP:
            long_candidate = _mark_chop(long_candidate, gap)
            short_candidate = _mark_chop(short_candidate, gap)

    # Prefer a direction that passes hard filters; otherwise fall back to best soft score for observation.
    long_ok = long_candidate.hard_filter_passed
    short_ok = short_candidate.hard_filter_passed
    if long_ok and not short_ok:
        return long_candidate
    if short_ok and not long_ok:
        return short_candidate
    return long_candidate if long_candidate.score >= short_candidate.score else short_candidate


def evaluate_observation_signal(
    *,
    symbol: str,
    trend_candles: Sequence[Candle],
    entry_candles: Sequence[Candle],
    expected_hold_hours: float = 2.0,
) -> ObservationSignal:
    candidate = evaluate_observation_candidate(
        symbol=symbol,
        trend_candles=trend_candles,
        entry_candles=entry_candles,
        expected_hold_hours=expected_hold_hours,
    )
    return ObservationSignal(
        symbol=candidate.symbol,
        direction=candidate.direction,
        score=candidate.score,
        level=candidate.level,
        current_price=candidate.current_price,
        reference_entry=candidate.trigger_price,
        stop_loss=candidate.stop_loss,
        take_profit_1=candidate.take_profit_1,
        take_profit_2=candidate.take_profit_2,
        expected_hold_hours=candidate.expected_hold_hours,
        reasons=candidate.reasons,
    )
