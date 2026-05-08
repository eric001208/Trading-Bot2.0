from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter backtest trades CSV by entry time (UTC).")
    p.add_argument("--src", required=True, help="Source CSV path (exported by this project).")
    p.add_argument("--dst", required=True, help="Destination CSV path.")
    p.add_argument("--days", type=int, default=90, help="Keep rows with entry time >= now-<days> (UTC).")
    p.add_argument(
        "--since-utc",
        default="",
        help="Optional ISO datetime (UTC). If provided, overrides --days. Example: 2026-02-07 00:00:00+00:00",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"src not found: {src}")

    if args.since_utc.strip():
        since = datetime.fromisoformat(args.since_utc.strip())
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        since = since.astimezone(UTC)
    else:
        since = datetime.now(UTC) - timedelta(days=max(0, int(args.days)))

    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("missing CSV header")
        for r in reader:
            t = (r.get("入场时间(UTC)") or "").strip()
            if not t:
                continue
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt = dt.astimezone(UTC)
            if dt >= since:
                rows.append(r)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"filtered_rows={len(rows)} since_utc={since.isoformat(sep=' ', timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

