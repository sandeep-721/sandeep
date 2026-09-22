from pathlib import Path
import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retrieval.hybrid_search import HybridSearch

app = FastAPI(title="Universal RAG")

rag = HybridSearch()

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


def require_api_key(x_api_key: str | None) -> None:
    if API_KEY is None:
        return

    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "universal-rag"
    }


@app.post("/search")
def search(
    request: SearchRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    results = rag.search(
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
