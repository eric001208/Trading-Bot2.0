from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send plain-text messages via Telegram Bot API (alerts only, no trading)."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send_text(self, text: str, *, disable_preview: bool = True) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError:
                body = r.text[:500] if r.text else ""
                logger.error("Telegram API %s: %s", r.status_code, body)
                raise
