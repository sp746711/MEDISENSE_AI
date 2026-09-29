"""Ollama offline LLM provider for MediSense AI."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_available(self) -> bool:
        base_url = (self.settings.ollama_base_url or "").rstrip("/")
        if not base_url:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
        base_url = (self.settings.ollama_base_url or "").rstrip("/")
        model = self.settings.ollama_model or "qwen3:4b"

        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 150,
                            "temperature": 0.2,
                        },
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("response", "").strip()
                    if content:
                        return {
                            "status": "ok",
                            "content": content,
                            "provider": "ollama",
                            "model": model,
                        }
                return {
                    "status": "unavailable",
                    "content": None,
                    "message": f"Ollama returned status {res.status_code}",
                }
        except Exception as exc:
            logger.warning("Ollama inference error: %s", exc)
            return {
                "status": "unavailable",
                "content": None,
                "message": f"Ollama request failed: {exc}",
            }
