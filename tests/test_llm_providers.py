import pytest

from generation.llm import create_llm_provider
from generation.providers.ollama import OllamaProvider
from generation.providers.openai_compatible import OpenAICompatibleProvider


def test_ollama_provider_selection(monkeypatch):
    monkeypatch.setenv("UNIVERSAL_RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("UNIVERSAL_RAG_LLM_MODEL", "qwen3")

    provider = create_llm_provider()

    try:
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "qwen3"
        assert provider.base_url == "http://127.0.0.1:11434/v1"
    finally:
        provider.close()


def test_openai_compatible_provider_selection(monkeypatch):
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_PROVIDER",
        "openai_compatible",
    )
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_MODEL",
        "test-model",
    )
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_BASE_URL",
        "http://127.0.0.1:9000/v1",
    )

    provider = create_llm_provider()

    try:
        assert isinstance(
            provider,
            OpenAICompatibleProvider,
        )
        assert provider.model_name == "test-model"
        assert provider.base_url == "http://127.0.0.1:9000/v1"
    finally:
        provider.close()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_PROVIDER",
        "does-not-exist",
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider()


def test_openai_provider_uses_openai_endpoint(monkeypatch):
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_PROVIDER",
        "openai",
    )
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_MODEL",
        "test-model",
    )

    provider = create_llm_provider()

    try:
        assert isinstance(
            provider,
            OpenAICompatibleProvider,
        )
        assert provider.model_name == "test-model"
        assert provider.base_url == "https://api.openai.com/v1"
    finally:
        provider.close()
