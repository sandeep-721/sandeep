import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_FILE = Path(__file__).resolve().parent / "llm.json"

DEFAULT_API_KEY_ENV = "UNIVERSAL_RAG_LLM_API_KEY"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = DEFAULT_API_KEY_ENV

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None

        return os.getenv(self.api_key_env)


def load_llm_config(path: Path | None = None) -> LLMConfig:
    config_path = path or CONFIG_FILE

    if not config_path.exists():
        raise FileNotFoundError(
            f"LLM configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data: dict[str, Any] = json.load(file)

    provider = str(data.get("provider", "")).strip()
    model = str(data.get("model", "")).strip()

    if not provider:
        raise ValueError("LLM config requires a provider.")

    if not model:
        raise ValueError("LLM config requires a model.")

    base_url = data.get("base_url")
    if base_url is not None:
        base_url = str(base_url).strip() or None

    api_key_env = data.get("api_key_env")

    if api_key_env is None or not str(api_key_env).strip():
        api_key_env = DEFAULT_API_KEY_ENV
    else:
        api_key_env = str(api_key_env).strip()

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def save_llm_config(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
    path: Path | None = None,
) -> LLMConfig:
    config_path = path or CONFIG_FILE

    provider = provider.strip().lower()
    model = model.strip()
    base_url = base_url.strip() if base_url else None

    if api_key_env:
        api_key_env = api_key_env.strip() or DEFAULT_API_KEY_ENV
    else:
        api_key_env = DEFAULT_API_KEY_ENV

    if not provider:
        raise ValueError("LLM provider is required.")

    if not model:
        raise ValueError("LLM model is required.")

    data = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
    }

    config_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )

    return load_llm_config(config_path)