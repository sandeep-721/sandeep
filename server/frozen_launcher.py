from __future__ import annotations

import os

import uvicorn


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from server.app import app

    host = os.getenv("UNIVERSAL_RAG_HOST", "127.0.0.1")
    port = int(os.getenv("UNIVERSAL_RAG_PORT", "8000"))
    api_key = os.getenv("UNIVERSAL_RAG_API_KEY")

    local_hosts = {"127.0.0.1", "localhost", "::1"}

    if host not in local_hosts and not api_key:
        raise RuntimeError(
            "UNIVERSAL_RAG_API_KEY must be set when "
            "UNIVERSAL_RAG_HOST is not a local-only address."
        )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
