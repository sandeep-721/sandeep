import os
from typing import Any

import httpx

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for APIs exposing an OpenAI-compatible HTTP interface.

    This allows Universal-RAG to connect to OpenAI-compatible servers
    without making the OpenAI SDK a core dependency.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model_name = model_name

        self.base_url = (
            base_url
            or os.getenv("UNIVERSAL_RAG_LLM_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("UNIVERSAL_RAG_LLM_API_KEY")
        )

        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    @staticmethod
    def _content_from_response(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []

        if not choices:
            raise RuntimeError(
                "OpenAI-compatible provider returned no choices."
            )

        message = choices[0].get("message") or {}
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)

            return "".join(parts).strip()

        return str(content).strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                "OpenAI-compatible provider request failed "
                f"with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        return self._content_from_response(
            response.json()
        )

    def list_models(self) -> list[str]:
        response = httpx.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                "Could not list models from provider: "
                f"HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        data = response.json()

        models = data.get("data") or []

        return [
            item["id"]
            for item in models
            if isinstance(item, dict) and item.get("id")
        ]

    def validate_connection(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
