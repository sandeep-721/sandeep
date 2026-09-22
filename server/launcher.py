from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
