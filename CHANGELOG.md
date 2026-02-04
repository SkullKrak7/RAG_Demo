# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2026-02-04

### Fixed
- Updated LLM generator to use ChatHuggingFace wrapper for HuggingFace Inference API compatibility
- Fixed "not supported for task text-generation" error by using conversational task format
- Added rank-bm25 dependency for hybrid retrieval BM25 component

### Changed
- Generator now wraps HuggingFaceEndpoint with ChatHuggingFace for proper chat format
- Updated tests to mock ChatHuggingFace wrapper correctly
- Improved test coverage to 96.19%

## [1.0.0] - 2026-02-04

### Added

#### Core RAG System
- Modular RAG architecture with clean separation of concerns
- Hybrid retrieval combining BM25 and vector search
- Cross-encoder reranking for improved relevance
- Streaming LLM responses via HuggingFace API
- Source attribution with citation tracking and relevance scores
- Langfuse observability integration for tracing and monitoring
- Vector store builder for PDF document ingestion
- Streamlit web interface for interactive queries

#### Enterprise Features
- RAGAS evaluation framework (faithfulness, relevancy, precision, recall)
- Performance monitoring with latency tracking and P95 metrics
- Query/response caching with semantic similarity matching
- Comprehensive API documentation with Sphinx
- Docker containerization with multi-stage builds
- Docker Compose for local development

#### Quality Infrastructure
- CI/CD pipeline with GitHub Actions
- Automated testing with 95%+ code coverage
- Code quality gates (Black, Bandit, coverage thresholds)
- Pre-commit hooks for local validation
- Integration and unit test suites

### Technical Specifications
- Python 3.11+ support
- 105 passing tests
- 95.66% test coverage
- Pinned dependency versions
- Security scanning with Bandit
- Type hints throughout codebase

### Documentation
- Production-ready README with quickstart guide
- Architecture documentation
- Deployment guides (local, Docker, cloud)
- API reference with code examples
- CI/CD workflow documentation
