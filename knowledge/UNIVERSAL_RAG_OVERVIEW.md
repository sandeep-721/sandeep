# Universal RAG — System Overview

## Purpose

Universal RAG is a project-aware retrieval system designed to give AI agents reliable access to relevant knowledge from software projects, source code, configuration, documentation, tests, and other supported project files.

Its primary role is to help an AI agent understand a project and find the information needed to answer questions or perform development tasks.

Universal RAG is the KNOW/FIND layer. Application-specific MCP servers remain the DO/CONTROL layer.

## High-Level Architecture

The system follows this retrieval pipeline:

Query
→ Dense Semantic Retrieval
→ Sparse Lexical Retrieval
→ Reciprocal Rank Fusion (RRF)
→ Candidate Pool
→ Cross-Encoder Reranking
→ Top Relevant Results
→ AI Agent Context

Dense retrieval provides semantic similarity.

Sparse retrieval provides lexical and identifier-sensitive matching.

RRF combines the rankings from dense and sparse retrieval.

The reranker evaluates the retrieved candidate documents against the original query and produces the final relevance ordering.

## Knowledge Sources

Universal RAG can index multiple categories of project information, including:

- Source code
- Configuration files
- Documentation
- Tests
- Structured project files
- Text files
- Supported programming languages
- Project metadata

The system classifies indexed sources so retrieval can distinguish project code, configuration, documentation, tests, RAG implementation code, and third-party content.

## Dense Retrieval

Dense retrieval uses semantic embeddings to find content that is conceptually related to a query even when the exact words are different.

The current embedding model is:

Qwen/Qwen3-Embedding-0.6B

Dense retrieval is useful for natural-language questions and conceptual searches.

## Sparse Retrieval

Sparse retrieval uses lexical matching through a BM25-based index.

The lexical index is designed to preserve important programming identifiers, filenames, symbols, paths, and technical terminology.

Sparse retrieval is particularly useful when a query contains exact or near-exact technical terms.

## Hybrid Retrieval

Universal RAG does not rely exclusively on dense or sparse retrieval.

Both retrieval systems independently produce candidate results.

Their rankings are combined using standard chunk-level Reciprocal Rank Fusion.

The current retrieval pipeline retrieves up to 100 results from each retriever and creates an RRF candidate pool of up to 75 documents before reranking.

## Reranking

The RRF candidate pool is passed to a cross-encoder reranker.

The current reranker model is:

Qwen/Qwen3-Reranker-0.6B

The reranker considers:

- Semantic relevance
- Lexical evidence
- Answer evidence
- Structural evidence
- RRF rank preservation

Structural evidence is given stronger influence for explicit identifier queries and reduced influence for broad natural-language questions.

This prevents exact code identifiers from overwhelming conceptual questions.

## Source Classification

Universal RAG distinguishes between several source categories.

Important categories include:

- rag_code
- rag_configuration
- rag_documentation
- rag_test
- project_code
- project_configuration
- project_documentation
- project_test
- third_party_code
- third_party_documentation

This classification allows retrieval and future filtering to understand where information originates.

## Configuration

Important retrieval and model configuration is stored in:

config/settings.py

Current models:

- Embedding: Qwen/Qwen3-Embedding-0.6B
- Reranker: Qwen/Qwen3-Reranker-0.6B

Current default chunk configuration:

- Chunk size: 800
- Chunk overlap: 120

The system currently uses CUDA for model inference.

## Vector Database

Universal RAG currently uses Qdrant as its vector database.

The local development database is stored under:

data/qdrant

Vector records contain the embedding together with metadata describing the source and document.

## Lexical Database

Universal RAG maintains a persistent BM25 lexical index under:

data/lexical

The lexical index is rebuilt from authoritative Qdrant payload metadata and document content.

## Chunking

Files are divided into retrieval-sized chunks before indexing.

The chunking system is file-type aware and supports different strategies for:

- Generic text
- Source code
- Documentation
- Structured data

Chunking allows large files to be retrieved at a useful level of granularity rather than treating an entire file as one retrieval document.

## Metadata

Each indexed document can contain metadata such as:

- Source
- Content type
- Software
- Software version
- Project
- Repository
- Branch
- Programming language
- Human language
- Path
- Symbol
- Symbol kind
- File hash
- Chunk index
- Chunk boundaries
- Source type
- Source version
- Tags

Metadata enables future project-aware, version-aware, and source-aware retrieval.

## Relationship With MCP

Universal RAG and MCP have different responsibilities.

Universal RAG:

KNOW / FIND

It retrieves information and provides context.

MCP:

DO / CONTROL

It allows an AI agent to interact with external applications and tools.

For example, an AI agent may use Universal RAG to find where a Unity component is implemented and then use Unity MCP to inspect or modify the running Unity project.

Universal RAG is not intended to replace application-specific MCP servers.

## Design Goals

The long-term system should be:

- Universal across software projects
- Project-aware
- Version-aware
- Incremental
- Reliable
- Hardware-aware
- Model-provider independent
- Extensible
- Efficient on available hardware
- Suitable for local development
- Suitable for future production deployment

## Retrieval Principle

Retrieval quality should be evaluated using real queries and measurable retrieval results.

The system must not assume that retrieval is correct simply because code executes successfully.

Evaluation should verify whether relevant information is:

1. Indexed
2. Retrieved by dense search
3. Retrieved by sparse search
4. Preserved by RRF
5. Correctly ranked by the reranker
6. Suitable for answering the user's question

Missing knowledge should be treated as a corpus/indexing problem rather than automatically compensating through increasingly aggressive ranking heuristics.

## Current Development Status

The retrieval engine currently contains:

- File ingestion
- File classification
- Chunking
- Metadata extraction
- Dense embeddings
- Qdrant vector storage
- BM25 sparse retrieval
- Hybrid dense + sparse retrieval
- RRF fusion
- Cross-encoder reranking
- Retrieval diagnostics
- Evaluation infrastructure

The knowledge directory is intended to contain authoritative documentation and future curated knowledge about Universal RAG itself.
