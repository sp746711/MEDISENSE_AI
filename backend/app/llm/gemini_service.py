"""Gemini LLM provider (fallback) for MediSense AI."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        return bool((self.settings.gemini_api_key or "").strip())

    def generate(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
        api_key = (self.settings.gemini_api_key or "").strip()
        if not api_key:
            return {
                "status": "unavailable",
                "content": None,
                "message": "Gemini API key not configured",
            }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post(url, json=body)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            content = parts[0].get("text", "").strip()
                            return {
                                "status": "ok",
                                "content": content,
                                "provider": "gemini",
                                "model": "gemini-2.5-flash",
                            }
                return {
                    "status": "unavailable",
                    "content": None,
                    "message": f"Gemini API returned HTTP {res.status_code}",
                }
        except Exception as exc:
            logger.warning("Gemini generation failed: %s", exc)
            return {
                "status": "unavailable",
                "content": None,
                "message": f"Gemini request error: {exc}",
            }
