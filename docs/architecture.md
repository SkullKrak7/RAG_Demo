# Architecture

## System Overview

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────┐
│  RAG Pipeline   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Hybrid│  │Reranker│
│Retriever│  └───┬───┘
└───┬──┘      │
    │         │
┌───▼─────────▼───┐
│  LLM Generator   │
└──────────────────┘
```

## Module Structure

### Core (`rag_demo/core/`)
- **config.py**: Pydantic configuration with validation
- **models.py**: Data models (Source, RAGResponse)
- **exceptions.py**: Custom exception hierarchy

### Retrieval (`rag_demo/retrieval/`)
- **retriever.py**: Hybrid retrieval (BM25 + vector)
- **reranker.py**: Cross-encoder reranking

### Generation (`rag_demo/generation/`)
- **generator.py**: LLM generation with streaming
- **formatter.py**: Response formatting with citations

### Pipeline (`rag_demo/pipeline/`)
- **pipeline.py**: End-to-end RAG orchestration

### Ingestion (`rag_demo/ingestion/`)
- **builder.py**: Document loading and vector store creation

### Observability (`rag_demo/observability/`)
- **tracer.py**: Langfuse integration for tracing

### Evaluation (`rag_demo/evaluation/`)
- **evaluator.py**: RAGAS metrics evaluation

## Data Flow

1. **Query Input** → User submits question via Streamlit UI
2. **Retrieval** → Hybrid retriever fetches relevant documents
3. **Reranking** → Cross-encoder scores and reorders results
4. **Generation** → LLM generates answer with context
5. **Formatting** → Response formatted with source citations
6. **Display** → Answer and sources shown in UI

## Key Design Decisions

### Hybrid Retrieval
Combines BM25 (keyword) and vector (semantic) search for better recall.

### Cross-Encoder Reranking
Improves precision by scoring query-document pairs.

### Streaming Responses
Provides real-time feedback for better UX.

### Modular Architecture
Each component is independently testable and replaceable.

### Optional Observability
Langfuse tracing can be enabled/disabled without code changes.
