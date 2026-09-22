from pathlib import Path

from config.settings import (
    EXCLUDED_DIRECTORIES,
    SUPPORTED_EXTENSIONS,
)


def is_supported_file(path: Path) -> bool:
    """
    Return True when the path is a supported regular file.
    """
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def is_excluded(path: Path) -> bool:
    """
    Return True when any path component belongs to the
    configured exclusion list.
    """
    return any(
        part in EXCLUDED_DIRECTORIES
        for part in path.parts
    )


def read_text_file(path: Path) -> str:
    """
    Read a text file as UTF-8.

    utf-8-sig handles both:
    - normal UTF-8 files
    - UTF-8 files containing a BOM

    When a BOM is present, Python removes it automatically
    instead of returning it as '\\ufeff' in the document text.

    errors='replace' prevents a single malformed byte from
    breaking the entire indexing pipeline.
    """
    return path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )


def discover_files(root: Path) -> list[Path]:
    """
    Recursively discover supported files while respecting
    the configured exclusion directories.
    """
    if not root.exists():
        return []

    return [
        path
        for path in root.rglob("*")
        if not is_excluded(path)
        and is_supported_file(path)
    ]