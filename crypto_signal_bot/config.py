from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_log_level(raw: str) -> str:
    level = raw.strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        return "INFO"
    return level


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            log_level=_parse_log_level(os.getenv("LOG_LEVEL", "INFO")),
        )

    def validate(self) -> None:
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not self.telegram_chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required")
