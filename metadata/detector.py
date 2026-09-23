from pathlib import Path


# ---------------------------------------------------------------------------
# Human-language detection
# ---------------------------------------------------------------------------

# Unicode ranges for common writing systems.
#
# These are intentionally script-level detectors rather than word-level
# language classifiers. This keeps detection fast, local, dependency-free,
# and reliable for scripts such as Telugu.
SCRIPT_RANGES = {
    "te": (
        0x0C00,
        0x0C7F,
    ),  # Telugu

    "ta": (
        0x0B80,
        0x0BFF,
    ),  # Tamil

    "hi": (
        0x0900,
        0x097F,
    ),  # Devanagari

    "kn": (
        0x0C80,
        0x0CFF,
    ),  # Kannada

    "ml": (
        0x0D00,
        0x0D7F,
    ),  # Malayalam

    "bn": (
        0x0980,
        0x09FF,
    ),  # Bengali

    "gu": (
        0x0A80,
        0x0AFF,
    ),  # Gujarati

    "pa": (
        0x0A00,
        0x0A7F,
    ),  # Gurmukhi

    "or": (
        0x0B00,
        0x0B7F,
    ),  # Odia

    "ar": (
        0x0600,
        0x06FF,
    ),  # Arabic

    "he": (
        0x0590,
        0x05FF,
    ),  # Hebrew

    "ru": (
        0x0400,
        0x04FF,
    ),  # Cyrillic

    "el": (
        0x0370,
        0x03FF,
    ),  # Greek

    "ja": (
        0x3040,
        0x30FF,
    ),  # Hiragana + Katakana

    "ko": (
        0xAC00,
        0xD7AF,
    ),  # Hangul

    "zh": (
        0x4E00,
        0x9FFF,
    ),  # CJK Unified Ideographs
}


def detect_human_language(
    text: str,
) -> str:
    """
    Detect the dominant human writing system in text.

    Returns an ISO 639-1 language code where the writing system
    provides a useful language signal.

    Latin-script text is intentionally returned as "en" only
    when English can be reasonably inferred from the available
    text. Otherwise "unknown" is returned.

    This function is script detection, not a full linguistic
    classifier.
    """

    if not text or not text.strip():
        return "unknown"

    counts = {
        language: 0
        for language in SCRIPT_RANGES
    }

    latin_count = 0

    for character in text:
        codepoint = ord(character)

        # Basic Latin / Latin-1 letters.
        if (
            ("A" <= character <= "Z")
            or ("a" <= character <= "z")
            or (0x00C0 <= codepoint <= 0x024F)
        ):
            latin_count += 1
            continue

        for language, (
            start,
            end,
        ) in SCRIPT_RANGES.items():

            if start <= codepoint <= end:
                counts[language] += 1
                break

    if counts:
        detected_language, detected_count = max(
            counts.items(),
            key=lambda item: item[1],
        )

        if detected_count > 0:
            return detected_language

    if latin_count > 0:
        return "en"

    return "unknown"


def detect_human_languages(
    text: str,
) -> list[str]:
    """
    Detect all significant human-language scripts present in text.

    This is useful for mixed documents such as:

        English + Telugu
        English + Tamil
        English + Hindi

    The returned list is ordered by character frequency.
    """

    if not text or not text.strip():
        return []

    counts = {
        language: 0
        for language in SCRIPT_RANGES
    }

    latin_count = 0

    for character in text:
        codepoint = ord(character)

        if (
            ("A" <= character <= "Z")
            or ("a" <= character <= "z")
            or (0x00C0 <= codepoint <= 0x024F)
        ):
            latin_count += 1
            continue

        for language, (
            start,
            end,
        ) in SCRIPT_RANGES.items():

            if start <= codepoint <= end:
                counts[language] += 1
                break

    if latin_count > 0:
        counts["en"] = latin_count

    return [
        language
        for language, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if count > 0
    ]


# ---------------------------------------------------------------------------
# Programming-language helpers
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGE_BY_EXTENSION = {
    ".cs": "csharp",
    ".py": "python",

    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",

    ".ts": "typescript",
    ".tsx": "typescript",

    ".c": "c",
    ".h": "c",

    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",

    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".lua": "lua",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".php": "php",
    ".dart": "dart",

    ".hlsl": "hlsl",
    ".shader": "hlsl",

    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
}


def detect_programming_language(
    path: Path,
) -> str | None:
    """
    Detect a programming language from a file extension.
    """

    return PROGRAMMING_LANGUAGE_BY_EXTENSION.get(
        path.suffix.lower()
    )


# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

def detect_project(root: Path) -> str | None:
    root = root.resolve()

    # Unity
    unity_project_name = (
        root
        / "ProjectSettings"
        / "ProjectSettings.asset"
    )

    if unity_project_name.exists():
        text = unity_project_name.read_text(
            encoding="utf-8",
            errors="replace",
        )

        for line in text.splitlines():
            if line.startswith("  productName:"):
                return line.split(":", 1)[1].strip()

    # Unreal
    uprojects = list(root.glob("*.uproject"))

    if uprojects:
        return uprojects[0].stem

    # Generic fallback
    if root.name:
        return root.name

    return None


def detect_software(root: Path) -> str | None:
    root = root.resolve()

    # Unity
    if (
        root
        / "ProjectSettings"
        / "ProjectVersion.txt"
    ).exists():
        return "Unity"

    # Unreal Engine
    if list(root.glob("*.uproject")):
        return "Unreal"

    # Maya
    if (
        (root / "maya").exists()
        or (root / "scripts" / "maya").exists()
    ):
        return "Maya"

    # Blender
    if (root / "blender_manifest.toml").exists():
        return "Blender"

    # Houdini
    if (root / "houdini.env").exists():
        return "Houdini"

    return None


def detect_software_version(
    root: Path,
    software: str | None,
) -> str | None:
    root = root.resolve()

    if software == "Unity":
        version_file = (
            root
            / "ProjectSettings"
            / "ProjectVersion.txt"
        )

        if version_file.exists():
            text = version_file.read_text(
                encoding="utf-8",
                errors="replace",
            )

            for line in text.splitlines():
                if line.startswith("m_EditorVersion:"):
                    return line.split(":", 1)[1].strip()

    return None


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

def detect_source_type(
    path: Path,
    project_root: Path | None = None,
) -> str:
    """
    Classify a source into a universal knowledge category.

    The classification is based on the file's location,
    project relationship, and filename.

    This does not remove files from the index.
    It only gives retrieval additional knowledge about
    what kind of source each file represents.
    """

    path = path.resolve()

    project_root = (
        project_root.resolve()
        if project_root
        else None
    )

    parts_lower = [
        part.lower()
        for part in path.parts
    ]

    filename = path.name.lower()
    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # Universal RAG project
    # --------------------------------------------------------

    if project_root is not None and path == project_root:
        return "unknown"

    if (
        project_root is not None
        and path.is_relative_to(project_root)
    ):
        # The project being indexed is the Universal RAG itself.
        if project_root.name.lower() == "universal-rag":
            relative_parts = [
                part.lower()
                for part in path.relative_to(
                    project_root
                ).parts
            ]

            if (
                "tests" in relative_parts
                or filename.startswith("test_")
                or filename.startswith("test.")
            ):
                return "rag_test"

            if "config" in relative_parts:
                return "rag_configuration"

            if (
                "readme" in filename
                or suffix in {".md", ".txt"}
            ):
                return "rag_documentation"

            return "rag_code"

    # --------------------------------------------------------
    # Third-party package detection
    # --------------------------------------------------------

    third_party_markers = {
        "reallusion",
        "packages",
        "packagecache",
        "library",
        "plugins",
        "thirdparty",
        "third_party",
    }

    if any(
        marker in parts_lower
        for marker in third_party_markers
    ):
        if suffix in {
            ".cs",
            ".py",
            ".shader",
            ".hlsl",
            ".cginc",
            ".cpp",
            ".h",
            ".hpp",
        }:
            return "third_party_code"

        return "third_party_documentation"

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    configuration_names = {
        "settings.py",
        "opencode.json",
        "package.json",
        "manifest.json",
        "projectsettings.asset",
        "projectversion.txt",
        "blender_manifest.toml",
        "houdini.env",
    }

    if filename in configuration_names:
        if project_root is not None:
            return "project_configuration"

        return "rag_configuration"

    configuration_extensions = {
        ".asmdef",
        ".json",
        ".yaml",
        ".yml",
        ".ini",
        ".toml",
        ".config",
    }

    if suffix in configuration_extensions:
        return "project_configuration"

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    test_markers = {
        "tests",
        "test",
        "testing",
    }

    if any(
        marker in parts_lower
        for marker in test_markers
    ):
        return "project_test"

    if (
        filename.startswith("test_")
        or filename.startswith("test.")
        or filename.endswith("_test.py")
        or filename.endswith("_test.cs")
    ):
        return "project_test"

    # --------------------------------------------------------
    # Historical / release documentation
    # --------------------------------------------------------

    historical_names = {
        "changelog.md",
        "changes.md",
        "history.md",
        "migration.md",
        "migrations.md",
        "release_notes.md",
        "release-notes.md",
        "roadmap.md",
        "todo.md",
    }

    if filename in historical_names:
        if project_root is not None:
            return "project_history"

        return "rag_history"

    # --------------------------------------------------------
    # Documentation
    # --------------------------------------------------------

    documentation_names = {
        "agents.md",
        "new_agent.md",
        ".agent.md",
        "master.md",
        "master_architecture_plan.md",
        "execution_plan.md",
        "project goal.md",
        "projectprogress.md",
        "structure_plan.md",
    }

    if filename in documentation_names:
        if project_root is not None:
            return "project_documentation"

        return "rag_documentation"

    if suffix in {
        ".md",
        ".txt",
    }:
        if project_root is not None:
            return "project_documentation"

        return "rag_documentation"

    # --------------------------------------------------------
    # Source code
    # --------------------------------------------------------

    code_extensions = {
        ".cs",
        ".py",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".shader",
        ".hlsl",
        ".cginc",
        ".glsl",
        ".js",
        ".ts",
    }

    if suffix in code_extensions:
        if project_root is not None:
            return "project_code"

        return "rag_code"

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    return "unknown"