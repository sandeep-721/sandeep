from pathlib import Path

from metadata.detector import detect_source_type


PROJECT_ROOT = Path(r"D:\AI\SubstancePainterMCP")


def test_project_history_detection():
    assert detect_source_type(
        PROJECT_ROOT / "CHANGELOG.md",
        PROJECT_ROOT,
    ) == "project_history"

    assert detect_source_type(
        PROJECT_ROOT / "docs" / "ROADMAP.md",
        PROJECT_ROOT,
    ) == "project_history"


def test_project_primary_source_detection():
    assert detect_source_type(
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT,
    ) == "project_documentation"

    assert detect_source_type(
        PROJECT_ROOT / "tests" / "test_server.py",
        PROJECT_ROOT,
    ) == "project_test"

    assert detect_source_type(
        PROJECT_ROOT / "src" / "substance_painter_mcp" / "server.py",
        PROJECT_ROOT,
    ) == "project_code"
