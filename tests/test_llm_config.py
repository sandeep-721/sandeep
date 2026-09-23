import json

import pytest

from config.llm_config import load_llm_config
from generation.providers.ollama import OllamaProvider
from generation.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from generation.llm import create_llm_provider


def test_default_llm_config():
    config = load_llm_config()

    assert config.provider == "local_transformers"
    assert config.model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.api_key_env == "UNIVERSAL_RAG_LLM_API_KEY"


def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_MODEL",
        "qwen3",
    )

    provider = create_llm_provider()

    try:
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "qwen3"
    finally:
        provider.close()


def test_openai_compatible_config(monkeypatch):
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
    monkeypatch.setenv(
        "UNIVERSAL_RAG_LLM_API_KEY",
        "test-key",
    )

    provider = create_llm_provider()

    try:
        assert isinstance(
            provider,
            OpenAICompatibleProvider,
        )
        assert provider.model_name == "test-model"
        assert provider.base_url == "http://127.0.0.1:9000/v1"
        assert provider.api_key == "test-key"
    finally:
        provider.close()


def test_bom_encoded_config_is_supported(tmp_path):
    config_path = tmp_path / "llm.json"

    config_path.write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": "qwen3",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_env": None,
            }
        ),
        encoding="utf-8-sig",
    )

    config = load_llm_config(config_path)

    assert config.provider == "ollama"
    assert config.model == "qwen3"
    assert config.base_url == "http://127.0.0.1:11434/v1"


def test_invalid_config_path_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_llm_config(tmp_path / "missing.json")
