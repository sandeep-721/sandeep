from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    text: str
    start: int
    end: int


CODE_EXTENSIONS = {
    ".cs",
    ".py",
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
    ".shader",
    ".hlsl",
    ".cginc",
    ".glsl",
    ".vert",
    ".frag",
    ".compute",
}

DOCUMENT_EXTENSIONS = {
    ".md",
    ".rst",
}

STRUCTURED_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".asmdef",
    ".uxml",
    ".uss",
}


def _generic_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        if end < text_length:
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            ]

            boundary = max(candidates)

            if boundary > start + (
                chunk_size // 2
            ):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    text=chunk,
                    start=start,
                    end=end,
                )
            )

        if end >= text_length:
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def _line_chunks(
    text: str,
    chunk_size: int,
) -> list[TextChunk]:
    """
    Preserve logical line boundaries while
    keeping chunks below the target size.
    """

    if not text.strip():
        return []

    chunks = []

    start = 0
    current_start = 0
    current_length = 0

    lines = text.splitlines(
        keepends=True
    )

    for line in lines:
        line_length = len(line)

        if (
            current_length > 0
            and current_length + line_length
            > chunk_size
        ):
            chunk = text[
                current_start:start
            ].strip()

            if chunk:
                chunks.append(
                    TextChunk(
                        text=chunk,
                        start=current_start,
                        end=start,
                    )
                )

            current_start = start
            current_length = 0

        current_length += line_length
        start += line_length

    if current_start < len(text):
        chunk = text[
            current_start:
        ].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    text=chunk,
                    start=current_start,
                    end=len(text),
                )
            )

    return chunks


def _code_chunks(
    text: str,
    chunk_size: int,
) -> list[TextChunk]:
    """
    Code-aware first pass.

    Prefer blank-line boundaries and logical
    code blocks before falling back to line chunks.
    """

    if not text.strip():
        return []

    sections = []
    section_start = 0

    lines = text.splitlines(
        keepends=True
    )

    position = 0

    for index, line in enumerate(lines):
        next_position = (
            position + len(line)
        )

        is_blank = not line.strip()

        next_is_blank = (
            index + 1 < len(lines)
            and not lines[index + 1].strip()
        )

        if (
            is_blank
            and next_is_blank
        ):
            section_end = next_position

            section = text[
                section_start:section_end
            ]

            if section.strip():
                sections.append(
                    (
                        section_start,
                        section_end,
                        section,
                    )
                )

            section_start = section_end

        position = next_position

    if section_start < len(text):
        sections.append(
            (
                section_start,
                len(text),
                text[section_start:],
            )
        )

    chunks = []

    for start, end, section in sections:
        if len(section) <= chunk_size:
            chunks.append(
                TextChunk(
                    text=section.strip(),
                    start=start,
                    end=end,
                )
            )
        else:
            smaller = _line_chunks(
                section,
                chunk_size,
            )

            for chunk in smaller:
                chunks.append(
                    TextChunk(
                        text=chunk.text,
                        start=start + chunk.start,
                        end=start + chunk.end,
                    )
                )

    return chunks


def _find_markdown_heading_level(
    text: str,
) -> int | None:
    """
    Determine the deepest heading level that
    appears in the Markdown document.

    The deepest level is used as the primary
    semantic retrieval boundary so parent sections
    do not duplicate their child sections.
    """

    levels = []

    heading_pattern = re.compile(
        r"(?m)^[ \t]*(#{1,6})[ \t]+.+?(?:\r?\n|$)"
    )

    for match in heading_pattern.finditer(text):
        levels.append(
            len(match.group(1))
        )

    if not levels:
        return None

    return max(levels)


def _markdown_sections(
    text: str,
) -> list[tuple[int, int, str]]:
    """
    Split Markdown into semantic retrieval sections.

    The deepest heading level present in the document
    becomes the primary section boundary.

    This prevents a parent section such as '# Overview'
    from being indexed together with child sections such
    as '## Purpose' and then indexing those child sections
    again separately.
    """

    if not text.strip():
        return []

    primary_level = (
        _find_markdown_heading_level(
            text
        )
    )

    if primary_level is None:
        return [
            (
                0,
                len(text),
                text,
            )
        ]

    heading_pattern = re.compile(
        rf"(?m)^[ \t]*#{{{primary_level}}}[ \t]+.+?(?:\r?\n|$)"
    )

    matches = list(
        heading_pattern.finditer(text)
    )

    if not matches:
        return [
            (
                0,
                len(text),
                text,
            )
        ]

    sections = []

    first_heading_start = matches[0].start()

    if text[:first_heading_start].strip():
        sections.append(
            (
                0,
                first_heading_start,
                text[:first_heading_start],
            )
        )

    for index, match in enumerate(matches):
        section_start = match.start()

        if index + 1 < len(matches):
            section_end = matches[
                index + 1
            ].start()
        else:
            section_end = len(text)

        section = text[
            section_start:section_end
        ]

        if section.strip():
            sections.append(
                (
                    section_start,
                    section_end,
                    section,
                )
            )

    return sections


def _split_document_section(
    text: str,
    start_offset: int,
    chunk_size: int,
) -> list[TextChunk]:
    """
    Split one semantic documentation section
    while preserving paragraph boundaries.

    The section heading remains attached to the
    first chunk of that section.
    """

    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [
            TextChunk(
                text=text.strip(),
                start=start_offset,
                end=start_offset + len(text),
            )
        ]

    paragraphs = []

    paragraph_pattern = re.compile(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        re.DOTALL,
    )

    for match in paragraph_pattern.finditer(text):
        paragraph = match.group(0)

        if paragraph.strip():
            paragraphs.append(
                (
                    match.start(),
                    match.end(),
                    paragraph,
                )
            )

    if not paragraphs:
        return _line_chunks(
            text,
            chunk_size,
        )

    chunks = []

    current_start = None
    current_end = None

    for paragraph_start, paragraph_end, paragraph in paragraphs:
        if current_start is None:
            current_start = paragraph_start
            current_end = paragraph_end
            continue

        candidate = text[
            current_start:paragraph_end
        ]

        if len(candidate.strip()) <= chunk_size:
            current_end = paragraph_end
            continue

        chunk_text = text[
            current_start:current_end
        ].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start=start_offset + current_start,
                    end=start_offset + current_end,
                )
            )

        current_start = paragraph_start
        current_end = paragraph_end

    if current_start is not None:
        chunk_text = text[
            current_start:current_end
        ].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start=start_offset + current_start,
                    end=start_offset + current_end,
                )
            )

    final_chunks = []

    for chunk in chunks:
        if len(chunk.text) <= chunk_size:
            final_chunks.append(
                chunk
            )
            continue

        smaller = _line_chunks(
            chunk.text,
            chunk_size,
        )

        for smaller_chunk in smaller:
            final_chunks.append(
                TextChunk(
                    text=smaller_chunk.text,
                    start=chunk.start
                    + smaller_chunk.start,
                    end=chunk.start
                    + smaller_chunk.end,
                )
            )

    return final_chunks


def _document_chunks(
    text: str,
    chunk_size: int,
) -> list[TextChunk]:
    """
    Documentation-aware semantic chunking.

    Markdown is divided using the deepest meaningful
    heading level. Related paragraphs remain together
    until the configured chunk size is reached.
    """

    if not text.strip():
        return []

    sections = _markdown_sections(
        text
    )

    chunks = []

    for start, end, section in sections:
        section_chunks = _split_document_section(
            text=section,
            start_offset=start,
            chunk_size=chunk_size,
        )

        chunks.extend(
            section_chunks
        )

    return chunks


def _structured_chunks(
    text: str,
    chunk_size: int,
) -> list[TextChunk]:
    """
    Structured/configuration files use
    line-aware chunking.
    """

    return _line_chunks(
        text,
        chunk_size,
    )


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
    extension: str | None = None,
) -> list[TextChunk]:
    """
    Universal intelligent chunking.

    extension:
        File extension such as '.cs', '.py',
        '.md', '.json', etc.

    If no extension is supplied, the generic
    chunker is used for backwards compatibility.
    """

    if not text.strip():
        return []

    if extension:
        extension = extension.lower()

    if extension in CODE_EXTENSIONS:
        return _code_chunks(
            text,
            chunk_size,
        )

    if extension in DOCUMENT_EXTENSIONS:
        return _document_chunks(
            text,
            chunk_size,
        )

    if extension in STRUCTURED_EXTENSIONS:
        return _structured_chunks(
            text,
            chunk_size,
        )

    return _generic_chunks(
        text,
        chunk_size,
        overlap,
    )