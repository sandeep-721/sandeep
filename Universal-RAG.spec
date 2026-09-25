from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(".").resolve().parent

hiddenimports = []
for package in (
    "server",
    "config",
    "embeddings",
    "ingestion",
    "metadata",
    "retrieval",
    "reranking",
    "generation",
):
    hiddenimports.extend(collect_submodules(package))

a = Analysis(
    ["server/frozen_launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "evaluation",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Universal-RAG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Universal-RAG",
)
