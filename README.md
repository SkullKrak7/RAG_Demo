# FSW RAG Demo: Production-Grade Friction Stir Welding Defect Analysis

[![CI Pipeline](https://github.com/SkullKrak7/RAG_Demo/actions/workflows/ci.yml/badge.svg)](https://github.com/SkullKrak7/RAG_Demo/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/SkullKrak7/RAG_Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Enterprise-grade Retrieval-Augmented Generation (RAG) system for intelligent defect analysis in Friction Stir Welding (FSW) processes.

## Features

- Hybrid Retrieval: BM25 + vector search with cross-encoder reranking
- Streaming Responses: Real-time LLM output with HuggingFace API
- Source Attribution: Full citation tracking with relevance scores
- Observability: Langfuse integration for tracing and monitoring
- Modular Architecture: Clean separation of concerns with 100% test coverage
- Production Ready: Error handling, validation, and graceful degradation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/SkullKrak7/RAG_Demo.git
cd RAG_Demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup git hooks
./setup-hooks.sh
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required: HF_TOKEN
# Optional: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### Build Vector Store

```bash
# Place FSW PDF documents in data/ directory
python build_vectorstore.py --pdf-dir ./data --output-dir ./vectorstore
```

### Run Application

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

## Architecture

### System Components

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

### Module Structure

```
rag_demo/
├── core/              # Configuration, models, exceptions
├── ingestion/         # Document loading and vectorstore building
├── retrieval/         # Hybrid retrieval and reranking
├── generation/        # LLM generation and response formatting
├── pipeline/          # End-to-end RAG orchestration
└── observability/     # Langfuse tracing integration
```

## Usage

### Basic Query

```python
from rag_demo.core.config import RAGConfig
from rag_demo.ingestion.builder import VectorStoreBuilder
from rag_demo.retrieval.retriever import HybridRetriever
from rag_demo.pipeline.pipeline import RAGPipeline

# Initialize
config = RAGConfig()
builder = VectorStoreBuilder(config)
vectorstore = builder.load_vectorstore()

# Create retriever
retriever = HybridRetriever(vectorstore, documents, config)

# Create pipeline
pipeline = RAGPipeline(retriever, config)

# Query
response = pipeline.query("What causes wormhole defects in FSW?")
print(response.answer)

for source in response.sources:
    print(f"- {source.doc_name} (Page {source.page_num})")
```

### Streaming Query

```python
for chunk in pipeline.stream_query("Explain FSW process parameters"):
    print(chunk, end="", flush=True)
```

### With Observability

```python
from rag_demo.observability.tracer import RAGTracer

config = RAGConfig(langfuse_enabled=True)
tracer = RAGTracer(config)
pipeline = RAGPipeline(retriever, config, tracer=tracer)

response = pipeline.query("What are common FSW defects?")

# Feedback
tracer.score_feedback(1.0, "user_feedback")
tracer.flush()
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `MODEL_NAME` | LLM model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `TEMPERATURE` | LLM temperature | `0.05` |
| `RETRIEVAL_K` | Documents to retrieve | `5` |
| `RERANK_TOP_K` | Documents after reranking | `3` |
| `CHUNK_SIZE` | Document chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `LANGFUSE_ENABLED` | Enable tracing | `false` |

See `.env.example` for complete configuration options.

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v --cov=rag_demo

# Specific module
pytest tests/unit/test_pipeline.py -v

# With coverage report
pytest tests/ --cov=rag_demo --cov-report=html
```

### Code Quality

```bash
# Format code
black rag_demo/ tests/

# Lint
pylint rag_demo/

# Type checking
mypy rag_demo/
```

## Project Status

### Completed Features

- Modular RAG architecture
- Hybrid retrieval (BM25 + vector)
- Cross-encoder reranking
- Streaming LLM responses
- Source attribution and citations
- Langfuse observability
- Vector store builder
- Streamlit UI
- 100% test coverage on core modules

### Roadmap

- CI/CD pipeline (GitHub Actions)
- Integration tests
- RAGAS evaluation framework
- Performance monitoring
- API documentation
- Docker deployment

## Technical Specifications

### Dependencies

- **LangChain**: 1.2.8
- **Streamlit**: 1.53.1
- **ChromaDB**: 1.4.1
- **Sentence Transformers**: 5.2.2
- **Langfuse**: 3.12.1
- **PyPDF**: 6.6.2

### Models

- **Embeddings**: `sentence-transformers/paraphrase-MiniLM-L3-v2` (384 dims)
- **LLM**: `meta-llama/Llama-3.1-8B-Instruct` (8B params)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Performance

- **Query Latency**: < 2s (with reranking)
- **Embedding Speed**: ~50ms per query
- **Vector Store Size**: ~500MB (5 documents)

## License

MIT License - Copyright (c) 2026 Sai Karthik Kagolanu

## Contact

- **Author**: Sai Karthik Kagolanu
- **GitHub**: [@SkullKrak7](https://github.com/SkullKrak7)
- **Project**: [RAG_Demo](https://github.com/SkullKrak7/RAG_Demo)
