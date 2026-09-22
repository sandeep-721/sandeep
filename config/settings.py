from pathlib import Path

# ============================================================
# Universal RAG - Base Configuration
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================
# Vector Database
# ============================================================

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

# ============================================================
# Models
# ============================================================

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Runtime device
DEVICE = "cuda"

# ============================================================
# Retrieval
# ============================================================

DENSE_TOP_K = 50
RERANK_TOP_K = 10

# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ============================================================
# Ingestion
# ============================================================

# File types that the Universal RAG reader can process.
#
# This list is intentionally broader than a single software
# package so the same ingestion system can support:
# Unity, Maya, Blender, Unreal, Houdini, Substance,
# general repositories, documentation, and configuration files.
#
# Software-specific intelligence will be added later without
# changing the basic reader architecture.

SUPPORTED_EXTENSIONS = {
    # C# / Unity
    ".cs",
    ".asmdef",
    ".uxml",
    ".uss",

    # Python / DCC tools
    ".py",

    # General programming / scripting
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".lua",
    ".rb",
    ".go",
    ".rs",

    # Shaders
    ".shader",
    ".hlsl",
    ".cginc",
    ".glsl",
    ".vert",
    ".frag",
    ".compute",

    # Configuration / structured data
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",

    # Documentation / text
    ".md",
    ".txt",
    ".rst",
    ".log",
}

# Directories that should never be indexed.
#
# These are normally generated, cached, binary, temporary,
# or dependency directories and provide little value to RAG.

EXCLUDED_DIRECTORIES = {
    # Python
    ".venv",
    "venv",
    "__pycache__",

    # Version control
    ".git",
    ".svn",
    ".hg",

    # JavaScript / Node
    "node_modules",

    # .NET / Unity generated
    "bin",
    "obj",

    # Unity generated directories
    "Library",
    "Temp",
    "Logs",
    "Build",
    "Builds",

    # Universal RAG internal data
    "qdrant",
    "models",
    "processed",
}