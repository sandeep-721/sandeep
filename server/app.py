from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from retrieval.hybrid_search import HybridSearch

app = FastAPI(title="Universal RAG")

rag = HybridSearch()

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    project: str | None = None
    software: str | None = None
    software_version: str | None = None
    content_type: str | None = None
    language: str | None = None
    human_language: str | None = None


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
def search(request: SearchRequest):
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
