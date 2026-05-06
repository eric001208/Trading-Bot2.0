from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from crypto_signal_bot.indicators import ma25, ma99, volume_ma
from crypto_signal_bot.models import Candle

MIN_TREND_BARS = 120
MIN_ENTRY_BARS = 120
SIDEWAYS_EXCEPTION_SCORE = 95
SIDEWAYS_SCORE_PENALTY = 5
MIN_POSITION_SPACE_R = 1.2
GOOD_POSITION_SPACE_R = 2.0


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
        return (
            f"交易观察备案：{self.symbol} {self.direction}\n\n"
            f"信号级别：{self.level} / {self.signal_grade}\n"
            f"综合评分：{self.score} / 100\n"
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


def _pct_change(candles: Sequence[Candle], lookback: int) -> float | None:
    if len(candles) <= lookback:
        return None
    old = candles[-lookback - 1].close
    if old <= 0:
        return None
    return (candles[-1].close - old) / old


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
    reasons: list[str] = []

    slope = _ma25_slope(trend_candles)
    ma_alignment = trend_ma25 > trend_ma99 if is_long else trend_ma25 < trend_ma99
    trend_ok = trend_last.close > trend_ma99 if is_long else trend_last.close < trend_ma99
    ma25_ok = trend_last.close > trend_ma25 if is_long else trend_last.close < trend_ma25
    slope_ok = slope > 0 if is_long else slope < 0

    if ma_alignment:
        score += 12
        reasons.append("1h MA25 与 MA99 排列支持当前方向。")
    if trend_ok:
        score += 12
        reasons.append("1h 价格位于 MA99 的有利方向，过滤掉一部分逆势交易。")
    if ma25_ok and slope_ok:
        score += 10
        reasons.append("1h 短中期方向和均线斜率一致。")

    recent = entry_candles[-16:-1]
    recent_range_high = max(c.high for c in recent)
    recent_range_low = min(c.low for c in recent)
    range_pct = (recent_range_high - recent_range_low) / entry_last.close

    if 0.003 <= range_pct <= 0.035:
        score += 8
        reasons.append("15m 近期波动区间适中，止损和目标更容易设计。")

    if adx >= 18:
        score += 8
        reasons.append("15m ADX 显示有一定趋势强度，减少纯震荡假突破。")
    elif adx < 12:
        reasons.append("15m ADX 偏低，市场偏震荡，候选质量降低。")

    if is_long:
        pulled_back = recent_range_low <= entry_ma25 * 1.004 or recent_range_low <= entry_ma99 * 1.01
        not_overheated = 42 <= rsi <= 68
        trigger = max(entry_last.high, recent_range_high) + 0.08 * atr
        trigger_gap = (trigger - entry_last.close) / entry_last.close
        above_range_mid = entry_last.close >= (recent_range_high + recent_range_low) / 2
        if pulled_back:
            score += 20
            reasons.append("15m 有回踩均线或结构区的动作，不是单纯追高。")
        if not_overheated:
            score += 10
            reasons.append("15m RSI 处于相对健康区间，暂未明显过热。")
        if entry_last.close >= entry_ma25:
            score += 8
            reasons.append("15m 当前价格已经重新靠近或站上 MA25。")
        if above_range_mid:
            score += 8
            reasons.append("15m 价格回到近期结构中轴上方，突破触发更有意义。")
    else:
        pulled_back = recent_range_high >= entry_ma25 * 0.996 or recent_range_high >= entry_ma99 * 0.99
        not_overheated = 32 <= rsi <= 58
        trigger = min(entry_last.low, recent_range_low) - 0.08 * atr
        trigger_gap = (entry_last.close - trigger) / entry_last.close
        below_range_mid = entry_last.close <= (recent_range_high + recent_range_low) / 2
        if pulled_back:
            score += 20
            reasons.append("15m 有反弹到均线或结构区的动作，不是单纯追空。")
        if not_overheated:
            score += 10
            reasons.append("15m RSI 处于相对健康区间，暂未明显超跌。")
        if entry_last.close <= entry_ma25:
            score += 8
            reasons.append("15m 当前价格已经重新靠近或跌回 MA25 下方。")
        if below_range_mid:
            score += 8
            reasons.append("15m 价格回到近期结构中轴下方，跌破触发更有意义。")

    if 0.0004 <= trigger_gap <= 0.006:
        score += 10
        reasons.append("触发价距离当前价格不远，适合短时间确认。")
    elif trigger_gap <= 0.012:
        score += 5
        reasons.append("触发价略远，需要等待更明确的突破确认。")
    else:
        reasons.append("触发价距离偏远，本轮候选质量降低。")

    if entry_last.volume >= 1.25 * vma:
        score += 12
        reasons.append("15m 成交量明显高于近20根均量，参与度较高。")
    elif entry_last.volume >= 0.9 * vma:
        score += 8
        reasons.append("15m 成交量不弱，具备基本确认。")

    stop, tp1, tp2 = _build_candidate_prices(
        direction=direction,
        current=entry_last.close,
        trigger=trigger,
        recent=entry_candles[-16:],
        atr=atr,
        target_rr=target_rr,
    )
    risk_pct = abs(trigger - stop) / trigger
    if 0.0025 <= risk_pct <= 0.014:
        score += 10
        reasons.append("止损距离相对合理，能支持更高盈亏比。")
    elif risk_pct <= 0.022:
        score += 5
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
    reasons.append(f"持仓时间因子：{hold_plan.detail}")
    final_score = _clamp_score(score)

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
    weekly_filter: bool = True,
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
    )
    candidate = long_candidate if long_candidate.score >= short_candidate.score else short_candidate

    if weekly_filter and candidate.direction in {"做多观察", "做空观察"}:
        weekly = classify_weekly_trend(trend_candles)
        candidate = _apply_long_context_filter(candidate, weekly, label="单币1-2周趋势")
        if candidate.level == "暂不关注":
            return candidate

    if market_candles is None or symbol == "BTCUSDT" or candidate.direction not in {"做多观察", "做空观察"}:
        return candidate

    market_trend = classify_weekly_trend(market_candles)
    candidate = _apply_long_context_filter(candidate, market_trend, label="BTC整体1-2周趋势")
    if candidate.level == "暂不关注":
        return candidate

    regime = classify_market_regime(market_candles)
    symbol_return = _pct_change(trend_candles, 6)
    btc_return = _pct_change(market_candles, 6)
    symbol_return_24h = _pct_change(trend_candles, 24)
    btc_return_24h = _pct_change(market_candles, 24)
    relative_reasons: tuple[str, ...] = ()
    if symbol_return is not None and btc_return is not None:
        relative = symbol_return - btc_return
        if candidate.direction == "做多观察" and relative < -0.003:
            return replace(
                candidate,
                score=min(candidate.score, 64),
                level="暂不关注",
                reasons=candidate.reasons
                + (
                    f"相对强弱过滤：近6小时本币相对 BTC 落后 {abs(relative) * 100:.2f}%，做多备案作废。",
                ),
            )
        if candidate.direction == "做空观察" and relative > 0.003:
            return replace(
                candidate,
                score=min(candidate.score, 64),
                level="暂不关注",
                reasons=candidate.reasons
                + (
                    f"相对强弱过滤：近6小时本币相对 BTC 更强 {relative * 100:.2f}%，做空备案作废。",
                ),
            )
        if candidate.direction == "做多观察" and relative > 0.002:
            relative_reasons = (f"相对强弱通过：近6小时本币强于 BTC {relative * 100:.2f}%。",)
        elif candidate.direction == "做空观察" and relative < -0.002:
            relative_reasons = (f"相对强弱通过：近6小时本币弱于 BTC {abs(relative) * 100:.2f}%。",)
        else:
            relative_reasons = ("相对强弱中性：本币与 BTC 表现差距不明显。",)

    if symbol_return_24h is not None and btc_return_24h is not None:
        relative_24h = symbol_return_24h - btc_return_24h
        if candidate.direction == "做多观察":
            if relative_24h <= -0.01:
                return replace(
                    candidate,
                    score=min(candidate.score, 69),
                    level="暂不关注",
                    reasons=candidate.reasons
                    + (
                        f"24h相对强弱过滤：本币近24小时弱于 BTC {abs(relative_24h) * 100:.2f}%，做多只记录。",
                    ),
                )
            if relative_24h >= 0.008:
                relative_reasons += (f"24h相对强弱通过：本币强于 BTC {relative_24h * 100:.2f}%。",)
        elif candidate.direction == "做空观察":
            if relative_24h >= 0.01:
                return replace(
                    candidate,
                    score=min(candidate.score, 69),
                    level="暂不关注",
                    reasons=candidate.reasons
                    + (
                        f"24h相对强弱过滤：本币近24小时强于 BTC {relative_24h * 100:.2f}%，做空只记录。",
                    ),
                )
            if relative_24h <= -0.008:
                relative_reasons += (f"24h相对强弱通过：本币弱于 BTC {abs(relative_24h) * 100:.2f}%。",)

    if candidate.direction == "做多观察" and not regime.allows_long:
        return replace(
            candidate,
            score=min(candidate.score, 59),
            level="暂不关注",
            reasons=candidate.reasons + (f"大盘过滤：{regime.detail} 非BTC做多备案作废。",),
        )
    if candidate.direction == "做空观察" and not regime.allows_short:
        return replace(
            candidate,
            score=min(candidate.score, 59),
            level="暂不关注",
            reasons=candidate.reasons + (f"大盘过滤：{regime.detail} 非BTC做空备案作废。",),
        )
    bonus = 5 if regime.direction in {"偏强", "偏弱"} else 0
    new_score = _clamp_score(candidate.score + bonus)
    return replace(
        candidate,
        score=new_score,
        level=_level(new_score),
        reasons=candidate.reasons + (f"大盘过滤通过：{regime.detail}",) + relative_reasons,
    )


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
