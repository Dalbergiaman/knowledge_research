# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Run a specific script
uv run python rag/milvus_insertdoc.py
```

No test suite is configured yet.

## Architecture

This is a RAG (Retrieval-Augmented Generation) system backed by Milvus vector database, targeting a Chinese-language knowledge base.

**Data flow:** raw text → Chinese tokenization (jieba) → embedding via Aliyun DashScope API → stored in Milvus collection

**Key modules:**
- `rag/milvus_config.py` — Centralizes all config: Milvus connection params, embedding model settings, collection schema, and index config. All other modules import from here.
- `rag/milvus_insertdoc.py` — Document ingestion pipeline: connects to Milvus, creates collection if needed, inserts embedded documents.
- `main.py` — Top-level entry point (currently a placeholder).

**Milvus collection schema** (`ai_knowledge_base`):
- `id` (auto-increment), `text`, `embedding` (1024-dim), `source`, `section`, `keywords`
- Index: IVF_FLAT / L2, TOP_K=10

**Embedding service:** Aliyun DashScope (`text-embedding-V4`) accessed via the OpenAI-compatible SDK. Requires `DASHSCOPE_API_KEY` in environment.

## Environment Variables

Copy `.env.example` (or create `.env`) with:
```
DASHSCOPE_API_KEY=...
MILVUS_HOST=localhost   # default
MILVUS_PORT=19530       # default
```
