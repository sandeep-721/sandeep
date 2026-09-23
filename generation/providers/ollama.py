import os

from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama provider using its OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            model_name=model_name,
            base_url=(
                base_url
                or os.getenv("UNIVERSAL_RAG_LLM_BASE_URL")
                or self.DEFAULT_BASE_URL
            ),
            api_key=None,
            timeout=timeout,
        )
