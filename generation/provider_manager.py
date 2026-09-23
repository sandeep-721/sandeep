import hashlib

from config.llm_config import LLMConfig, load_llm_config
from generation.llm import create_llm_provider
from generation.providers.base import LLMProvider


class LLMProviderManager:
    """Lazily load and reuse the user's selected LLM provider."""

    def __init__(self):
        self._provider: LLMProvider | None = None
        self._signature: tuple | None = None
        self._config: LLMConfig | None = None

    @staticmethod
    def _signature_for(config: LLMConfig) -> tuple:
        api_key_digest = hashlib.sha256(
            (config.api_key or "").encode("utf-8")
        ).hexdigest()

        return (
            config.provider,
            config.model,
            config.base_url,
            config.api_key_env,
            api_key_digest,
        )

    def _ensure_provider(self) -> LLMProvider:
        config = load_llm_config()
        signature = self._signature_for(config)

        if self._provider is not None and signature == self._signature:
            return self._provider

        self.close()

        self._provider = create_llm_provider(
            provider=config.provider,
            model=config.model,
        )

        self._signature = signature
        self._config = config

        return self._provider

    def get_config(self) -> LLMConfig:
        self._ensure_provider()
        return self._config

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        provider = self._ensure_provider()

        return provider.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def list_models(self) -> list[str]:
        provider = self._ensure_provider()
        return provider.list_models()

    def validate_connection(self) -> bool:
        provider = self._ensure_provider()
        return provider.validate_connection()

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()

        self._provider = None
        self._signature = None
        self._config = None
