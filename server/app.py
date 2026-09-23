from pathlib import Path
import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config.llm_config import (
    load_llm_config,
    save_llm_config,
)
from generation.llm import (
    DEFAULT_MODELS,
    create_llm_provider,
)
from rag import RAG
from retrieval.hybrid_search import HybridSearch


app = FastAPI(title="Universal RAG")

search_engine = HybridSearch()
answer_engine = RAG()

STATIC_DIR = Path(__file__).parent / "static"
API_KEY = os.getenv("UNIVERSAL_RAG_API_KEY")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)
    project: str | None = None
    software: str | None = None
    software_version: str | None = None
    content_type: str | None = None
    language: str | None = None
    human_language: str | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    project: str | None = None


class LLMConfigRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=500)
    base_url: str | None = Field(default=None, max_length=1000)
    api_key_env: str | None = Field(default=None, max_length=200)


class LLMTestRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=500)
    base_url: str | None = Field(default=None, max_length=1000)
    api_key_env: str | None = Field(default=None, max_length=200)


LLM_PROVIDERS = [
    {
        "id": "local_transformers",
        "name": "Local Transformers",
        "local": True,
        "requires_api_key": False,
        "default_model": "",
        "default_base_url": None,
    },
    {
        "id": "ollama",
        "name": "Ollama",
        "local": True,
        "requires_api_key": False,
        "default_model": DEFAULT_MODELS["ollama"],
        "default_base_url": "http://127.0.0.1:11434/v1",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "local": False,
        "requires_api_key": True,
        "default_model": DEFAULT_MODELS["openai"],
        "default_base_url": "https://api.openai.com/v1",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "local": False,
        "requires_api_key": True,
        "default_model": "",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    {
        "id": "lmstudio",
        "name": "LM Studio",
        "local": True,
        "requires_api_key": False,
        "default_model": "",
        "default_base_url": "http://127.0.0.1:1234/v1",
    },
    {
        "id": "vllm",
        "name": "vLLM",
        "local": True,
        "requires_api_key": False,
        "default_model": "",
        "default_base_url": "http://127.0.0.1:8000/v1",
    },
    {
        "id": "openai_compatible",
        "name": "OpenAI-Compatible",
        "local": False,
        "requires_api_key": False,
        "default_model": "",
        "default_base_url": "http://127.0.0.1:11434/v1",
    },
    {
        "id": "custom",
        "name": "Custom Endpoint",
        "local": False,
        "requires_api_key": False,
        "default_model": "",
        "default_base_url": "",
    },
]


def require_api_key(x_api_key: str | None) -> None:
    if API_KEY is None:
        return

    if x_api_key is None or not secrets.compare_digest(
        x_api_key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def validate_provider_id(provider: str) -> str:
    provider_name = provider.strip().lower()

    allowed = {
        provider["id"]
        for provider in LLM_PROVIDERS
    }

    if provider_name not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider_name}",
        )

    return provider_name


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "universal-rag",
    }


@app.get("/llm/providers")
def llm_providers(
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    return {
        "providers": LLM_PROVIDERS,
    }


@app.get("/llm/config")
def llm_config(
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    config = load_llm_config()

    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "api_key_configured": bool(config.api_key),
    }


@app.post("/llm/config")
def update_llm_config(
    request: LLMConfigRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    provider = validate_provider_id(request.provider)

    config = save_llm_config(
        provider=provider,
        model=request.model,
        base_url=request.base_url,
        api_key_env=request.api_key_env,
    )

    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "api_key_configured": bool(config.api_key),
    }


@app.get("/llm/test")
def test_saved_llm_connection(
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    try:
        connected = answer_engine.llm.validate_connection()
        models = answer_engine.llm.list_models()
        config = answer_engine.llm.get_config()

        return {
            "connected": bool(connected),
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "models": models,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM connection test failed: {exc}",
        ) from exc


@app.post("/llm/test")
def test_temporary_llm_connection(
    request: LLMTestRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    provider_name = validate_provider_id(request.provider)

    api_key = ""

    if request.api_key_env:
        api_key = os.getenv(
            request.api_key_env.strip(),
            "",
        )

    provider = None

    try:
        provider = create_llm_provider(
            provider=provider_name,
            model=request.model.strip(),
            base_url=request.base_url,
            api_key=api_key,
        )

        connected = provider.validate_connection()
        models = provider.list_models()

        return {
            "connected": bool(connected),
            "provider": provider_name,
            "model": request.model.strip(),
            "base_url": request.base_url,
            "api_key_configured": bool(api_key),
            "models": models,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Temporary LLM connection test failed: {exc}",
        ) from exc

    finally:
        if provider is not None:
            provider.close()


@app.get("/llm/models")
def llm_models(
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    try:
        models = answer_engine.llm.list_models()
        config = answer_engine.llm.get_config()

        return {
            "provider": config.provider,
            "models": models,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load models: {exc}",
        ) from exc


@app.post("/search")
def search(
    request: SearchRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    results = search_engine.search(
        query=request.query,
        limit=request.limit,
        project=request.project,
        software=request.software,
        software_version=request.software_version,
        content_type=request.content_type,
        language=request.language,
        human_language=request.human_language,
    )

    return {
        "query": request.query,
        "results": results,
    }


@app.post("/ask")
def ask(
    request: AskRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    try:
        result = answer_engine.ask(
            question=request.query,
            project=request.project,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM generation failed: {exc}",
        ) from exc

    config = answer_engine.llm.get_config()

    return {
        "query": request.query,
        "project": request.project,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "results": result.get("results", []),
        "model": {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
        },
    }