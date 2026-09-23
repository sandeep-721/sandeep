import os

from .providers.base import LLMProvider
from .providers.local_transformers import TransformersLocalProvider
from .providers.ollama import OllamaProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from config.llm_config import load_llm_config


DEFAULT_PROVIDER = "local_transformers"


DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "ollama": "qwen3",
    "openrouter": "",
    "lmstudio": "",
    "vllm": "",
    "openai_compatible": "",
    "custom": "",
}


def create_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """
    Create a Universal-RAG LLM provider.

    Explicit arguments have priority over environment variables
    and saved configuration. This allows the application to test
    temporary provider settings without changing the saved config.
    """

    config = load_llm_config()

    provider_name = (
        provider
        or os.getenv("UNIVERSAL_RAG_LLM_PROVIDER")
        or config.provider
        or DEFAULT_PROVIDER
    ).strip().lower()

    configured_model = (
        model
        or os.getenv("UNIVERSAL_RAG_LLM_MODEL")
        or config.model
        or DEFAULT_MODELS.get(provider_name, "")
    ).strip()

    configured_base_url = (
        base_url
        or os.getenv("UNIVERSAL_RAG_LLM_BASE_URL")
        or config.base_url
    )

    configured_api_key = (
        api_key
        if api_key is not None
        else (
            os.getenv("UNIVERSAL_RAG_LLM_API_KEY")
            or config.api_key
        )
    )

    if provider_name in {
        "local",
        "local_transformers",
        "transformers",
        "huggingface",
    }:
        return TransformersLocalProvider(
            model_name=(
                configured_model
                or TransformersLocalProvider.DEFAULT_MODEL
            )
        )

    if provider_name == "ollama":
        if not configured_model:
            raise ValueError(
                "A model must be configured for the Ollama provider."
            )

        return OllamaProvider(
            model_name=configured_model,
            base_url=(
                configured_base_url
                or OllamaProvider.DEFAULT_BASE_URL
            ),
        )

    if provider_name in {
        "openai",
        "openrouter",
        "lmstudio",
        "vllm",
        "openai_compatible",
        "custom",
    }:
        if not configured_model:
            raise ValueError(
                "A model must be configured for "
                f"the {provider_name} provider."
            )

        default_base_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "lmstudio": "http://127.0.0.1:1234/v1",
            "vllm": "http://127.0.0.1:8000/v1",
        }

        resolved_base_url = (
            configured_base_url
            or default_base_urls.get(
                provider_name,
                OpenAICompatibleProvider.DEFAULT_BASE_URL,
            )
        )

        return OpenAICompatibleProvider(
            model_name=configured_model,
            base_url=resolved_base_url,
            api_key=configured_api_key,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}. "
        "Supported providers: local_transformers, "
        "ollama, openai, openrouter, lmstudio, vllm, "
        "openai_compatible, custom."
    )


# Backward compatibility.
LocalLLM = TransformersLocalProvider