import os

import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Universal RAG")

API_URL = os.getenv("UNIVERSAL_RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    project: str | None = None,
    software: str | None = None,
    software_version: str | None = None,
    content_type: str | None = None,
    language: str | None = None,
    human_language: str | None = None,
) -> dict:
    """Search the Universal RAG knowledge base."""

    payload = {
        "query": query,
        "limit": limit,
        "project": project,
        "software": software,
        "software_version": software_version,
        "content_type": content_type,
        "language": language,
        "human_language": human_language,
    }

    response = httpx.post(
        f"{API_URL}/search",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
