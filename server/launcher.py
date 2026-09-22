from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    host = os.getenv("UNIVERSAL_RAG_HOST", "127.0.0.1")
    port = os.getenv("UNIVERSAL_RAG_PORT", "8000")
    api_key = os.getenv("UNIVERSAL_RAG_API_KEY")

    local_hosts = {"127.0.0.1", "localhost", "::1"}

    if host not in local_hosts and not api_key:
        raise RuntimeError(
            "UNIVERSAL_RAG_API_KEY must be set when "
            "UNIVERSAL_RAG_HOST is not a local-only address."
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            host,
            "--port",
            port,
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
