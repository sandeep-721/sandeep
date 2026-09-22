import hashlib
import uuid
from pathlib import Path

from ingestion.reader import (
    discover_files,
    read_text_file,
)
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from metadata.schema import DocumentMetadata
from metadata.detector import (
    detect_project,
    detect_software,
    detect_software_version,
    detect_human_language,
    detect_source_type,
)
from metadata.symbols import detect_symbols
from retrieval.vector_store import VectorStore
from retrieval.lexical_index import LexicalIndex


LANGUAGE_MAP = {
    '.cs': 'csharp',
    '.asmdef': 'json',
    '.uxml': 'xml',
    '.uss': 'uss',
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.java': 'java',
    '.lua': 'lua',
    '.rb': 'ruby',
    '.go': 'go',
    '.rs': 'rust',
    '.shader': 'unity-shader',
    '.hlsl': 'hlsl',
    '.cginc': 'hlsl',
    '.glsl': 'glsl',
    '.vert': 'glsl',
    '.frag': 'glsl',
    '.compute': 'hlsl',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.xml': 'xml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.cfg': 'config',
    '.conf': 'config',
    '.md': 'markdown',
    '.txt': 'text',
    '.rst': 'restructuredtext',
    '.log': 'log',
}


CODE_EXTENSIONS = {
    '.cs',
    '.py',
    '.js',
    '.jsx',
    '.ts',
    '.tsx',
    '.java',
    '.cpp',
    '.c',
    '.h',
    '.hpp',
    '.lua',
    '.rb',
    '.go',
    '.rs',
    '.shader',
    '.hlsl',
    '.cginc',
    '.glsl',
    '.vert',
    '.frag',
    '.compute',
}


def calculate_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open('rb') as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b'',
        ):
            hasher.update(block)

    return hasher.hexdigest()


def detect_content_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in CODE_EXTENSIONS:
        return 'code'

    if suffix in {
        '.md',
        '.txt',
        '.rst',
        '.log',
    }:
        return 'documentation'

    if suffix in {
        '.json',
        '.yaml',
        '.yml',
        '.xml',
        '.toml',
        '.ini',
        '.cfg',
        '.conf',
        '.asmdef',
        '.uxml',
        '.uss',
    }:
        return 'configuration'

    return 'text'


def find_enclosing_symbol(
    symbols: list[dict],
    chunk_start: int,
) -> dict | None:
    """
    Find the most recent symbol declared before the chunk.

    Symbols are detected from the complete source file,
    not from individual chunks. This allows a chunk to retain
    the identity of the class or method that owns it even
    when the declaration itself is in an earlier chunk.
    """

    if not symbols:
        return None

    candidates = [
        symbol
        for symbol in symbols
        if symbol.start <= chunk_start
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda symbol: symbol.start,
    )


def build_symbol_metadata(
    symbols,
    chunk_start: int,
    chunk_end: int,
) -> dict:
    """
    Build canonical symbol metadata for a chunk.

    The metadata is intentionally language-independent.
    """

    metadata = {
        'symbol': None,
        'symbol_kind': None,
        'class_name': None,
        'member_name': None,
        'member_kind': None,
        'chunk_type': 'code',
    }

    if not symbols:
        return metadata

    selected = None

    containing = [
        symbol
        for symbol in symbols
        if symbol.has_scope
        and symbol.scope_start <= chunk_start
        and symbol.scope_end >= chunk_end
    ]

    if containing:
        selected = min(
            containing,
            key=lambda symbol: (
                symbol.scope_end - symbol.scope_start
            )
        )

    if selected is None:
        selected = find_enclosing_symbol(
            symbols,
            chunk_start,
        )

    if selected is None:
        return metadata

    selected_kind = selected.kind

    metadata['symbol'] = selected.name
    metadata['symbol_kind'] = selected_kind

    if selected_kind in {
        'class',
        'struct',
        'interface',
        'enum',
        'record',
        'trait',
        'module',
        'namespace',
        'object',
    }:
        metadata['class_name'] = selected.name
        metadata['chunk_type'] = selected_kind

    elif selected_kind in {
        'method',
        'function',
        'constructor',
        'destructor',
    }:
        metadata['member_name'] = selected.name
        metadata['member_kind'] = selected_kind
        metadata['chunk_type'] = selected_kind

        if selected.parent_kind in {
            'class',
            'struct',
            'interface',
            'enum',
            'record',
            'trait',
            'module',
            'namespace',
            'object',
        }:
            metadata['class_name'] = selected.parent_symbol

    else:
        metadata['chunk_type'] = selected_kind

    return metadata


class Indexer:
    def __init__(self, project_root: Path | None = None):
        self.embedder = None
        self.store = VectorStore()
        self.lexical_index = LexicalIndex()

        self.project_root = (
            project_root.resolve()
            if project_root
            else None
        )

        self.software = None
        self.software_version = None
        self.project = None

        if self.project_root:
            self.software = detect_software(
                self.project_root
            )

            self.software_version = detect_software_version(
                self.project_root,
                self.software,
            )

            self.project = detect_project(
                self.project_root
            )

    @staticmethod
    def make_chunk_id(
        path: Path,
        chunk_index: int,
    ) -> str:
        identity = f'{path.resolve()}::{chunk_index}'

        hash_value = hashlib.sha256(
            identity.encode('utf-8')
        ).hexdigest()[:32]

        return str(uuid.UUID(hash_value))

    def index_file(
        self,
        path: Path,
        software: str | None = None,
        software_version: str | None = None,
        project: str | None = None,
    ):
        path = path.resolve()

        file_hash = calculate_file_hash(path)

        if self.store.file_exists(file_hash):
            print(f'Skipped unchanged: {path}')

            return {
                'status': 'unchanged',
                'chunks': 0,
            }

        source = str(path)

        self.store.delete_file(source)
        self.lexical_index.delete_file(source)

        text = read_text_file(path)

        # Detect natural/human language from the complete source file.
        # This is intentionally separate from the programming/document
        # format stored in the `language` field.
        human_language = detect_human_language(text)

        content_type = detect_content_type(path)

        language = LANGUAGE_MAP.get(
            path.suffix.lower(),
            'unknown',
        )

        # Classify the source independently from content type.
        #
        # Examples:
        #   rag_code
        #   rag_test
        #   rag_configuration
        #   rag_documentation
        #   project_code
        #   project_test
        #   project_configuration
        #   project_documentation
        #   third_party_code
        #   third_party_documentation
        source_type = detect_source_type(
            path,
            self.project_root,
        )

        file_symbols = []

        if content_type == 'code':
            file_symbols = detect_symbols(
                text,
                path.suffix.lower(),
            )

        chunks = chunk_text(
            text,
            extension=path.suffix.lower(),
        )

        if not chunks:
            return {
                'status': 'empty',
                'chunks': 0,
            }

        if self.embedder is None:
            self.embedder = Embedder()

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = self.embedder.encode(texts)

        ids = [
            self.make_chunk_id(
                path,
                index,
            )
            for index in range(len(chunks))
        ]

        payloads = []

        for index, chunk in enumerate(chunks):

            symbol_metadata = {
                'symbol': None,
                'symbol_kind': None,
                'class_name': None,
                'member_name': None,
                'member_kind': None,
                'chunk_type': content_type,
            }

            if content_type == 'code':
                symbol_metadata = build_symbol_metadata(
                    symbols=file_symbols,
                    chunk_start=chunk.start,
                    chunk_end=chunk.end,
                )

            metadata = DocumentMetadata(
                source=source,
                content_type=content_type,

                software=software,
                software_version=software_version,

                project=project,

                language=language,
                human_language=human_language,

                path=source,

                symbol=symbol_metadata['symbol'],
                symbol_kind=symbol_metadata['symbol_kind'],

                file_hash=file_hash,

                chunk_index=index,
                chunk_start=chunk.start,
                chunk_end=chunk.end,

                source_type=source_type,
            )

            payload = metadata.to_dict()

            payload['text'] = chunk.text

            # Canonical code hierarchy metadata.
            payload['class_name'] = (
                symbol_metadata['class_name']
            )

            payload['member_name'] = (
                symbol_metadata['member_name']
            )

            payload['member_kind'] = (
                symbol_metadata['member_kind']
            )

            payload['chunk_type'] = (
                symbol_metadata['chunk_type']
            )

            payloads.append(payload)

        self.store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

        for point_id, payload in zip(ids, payloads):
            self.lexical_index.upsert(
                point_id,
                payload,
            )

        self.lexical_index.save()

        return {
            'status': 'indexed',
            'chunks': len(chunks),
            'file_hash': file_hash,
            'human_language': human_language,
            'source_type': source_type,
        }

    def index_project(self):
        if not self.project_root:
            raise ValueError(
                'project_root is required for project indexing'
            )

        files = discover_files(
            self.project_root
        )

        results = []

        for path in files:
            result = self.index_file(
                path,
                software=self.software,
                software_version=self.software_version,
                project=self.project,
            )

            results.append(
                {
                    'path': str(path),
                    **result,
                }
            )

        return results