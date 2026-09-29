"""Groq-hosted open model provider for MediSense AI."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GroqService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        return bool((self.settings.groq_api_key or "").strip())

    def generate(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
        api_key = (self.settings.groq_api_key or "").strip()
        if not api_key:
            return {
                "status": "unavailable",
                "content": None,
                "message": "Groq API key not configured",
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        return {
                            "status": "ok",
                            "content": content,
                            "provider": "groq",
                            "model": data.get("model", "llama-3.3-70b-versatile"),
                        }
                return {
                    "status": "unavailable",
                    "content": None,
                    "message": f"Groq API returned HTTP {res.status_code}",
                }
        except Exception as exc:
            logger.warning("Groq generation failed: %s", exc)
            return {
                "status": "unavailable",
                "content": None,
                "message": f"Groq request error: {exc}",
            }
