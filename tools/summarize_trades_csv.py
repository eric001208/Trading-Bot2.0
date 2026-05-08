from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Summary:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_return_pct: float = 0.0
    outcomes: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.outcomes is None:
            self.outcomes = Counter()

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades else 0.0

    @property
    def avg_return_pct(self) -> float:
        return (self.total_return_pct / self.trades) if self.trades else 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize exported backtest trades CSV.")
    p.add_argument("--csv", required=True, help="Trades CSV path exported by this project.")
    return p.parse_args()


def _profit_factor(pnls: list[float]) -> float:
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
    return mdd


def main() -> int:
    # Windows PowerShell may default to a legacy code page; avoid UnicodeEncodeError for Chinese text.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    by_symbol: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            symbol = (r.get("交易对") or "").strip().upper()
            if not symbol:
                continue
            by_symbol[symbol].append(r)

    if not by_symbol:
        raise SystemExit("no rows parsed from CSV")

    for symbol in sorted(by_symbol):
        rows = by_symbol[symbol]
        pnls: list[float] = []
        summary = Summary(trades=len(rows))
        for r in rows:
            try:
                pnl = float((r.get("收益率(%)") or "0").strip())
            except ValueError:
                pnl = 0.0
            pnls.append(pnl)
            summary.total_return_pct += pnl
            if pnl > 0:
                summary.wins += 1
            elif pnl < 0:
                summary.losses += 1
            outcome = (r.get("出场原因") or "").strip()
            if outcome:
                summary.outcomes[ outcome ] += 1

        pf = _profit_factor(pnls)
        mdd = _max_drawdown(pnls)
        pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
        top_outcomes = ", ".join(f"{k}:{v}" for k, v in summary.outcomes.most_common(6))
        print(
            f"{symbol}: trades={summary.trades} win_rate={summary.win_rate*100:.2f}% "
            f"total_return={summary.total_return_pct:.2f}% avg={summary.avg_return_pct:.3f}% "
            f"profit_factor={pf_text} max_drawdown={mdd:.2f}% outcomes=[{top_outcomes}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
