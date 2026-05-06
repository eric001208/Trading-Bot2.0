from __future__ import annotations

import logging
import sys


def _prefer_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent safe to call again)."""
    _prefer_utf8_streams()
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric)
        for h in root.handlers:
            h.setLevel(numeric)
        return

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
