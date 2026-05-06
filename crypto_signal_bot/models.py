from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass(frozen=True)
class UsdmTicker:
    symbol: str
    last_price: float
    quote_volume: float
    price_change_pct: float = 0.0


@dataclass(frozen=True)
class SignalResult:
    name: str
    direction: str
    score: int  # 0-100
    detail: str


@dataclass(frozen=True)
class Candle:
    """OHLCV bar; is_closed=False while the exchange is still updating the current kline."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    is_closed: bool = True

    @classmethod
    def from_kline(cls, k: Kline) -> Candle:
        return cls(
            open_time=k.open_time,
            open=k.open,
            high=k.high,
            low=k.low,
            close=k.close,
            volume=k.volume,
            close_time=k.close_time,
            is_closed=True,
        )

    def to_closed_copy(self) -> Candle:
        if self.is_closed:
            return self
        return Candle(
            open_time=self.open_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            close_time=self.close_time,
            is_closed=True,
        )
