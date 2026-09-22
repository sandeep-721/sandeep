from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Virtual environment Python not found: {PYTHON}")

    subprocess.run(
        [str(PYTHON), "-m", "server.mcp_server"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
