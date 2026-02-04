# Quick Start

## Installation

```bash
git clone https://github.com/SkullKrak7/RAG_Demo.git
cd RAG_Demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
./setup-hooks.sh
```

## Configuration

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
```

## Build Vector Store

```bash
python build_vectorstore.py --pdf-dir ./data --output-dir ./vectorstore
```

## Run Application

```bash
streamlit run app.py
```

## Basic Usage

```python
from rag_demo.core.config import RAGConfig
from rag_demo.ingestion.builder import VectorStoreBuilder
from rag_demo.retrieval.retriever import HybridRetriever
from rag_demo.pipeline.pipeline import RAGPipeline

# Initialize
config = RAGConfig()
builder = VectorStoreBuilder(config)
vectorstore, documents = builder.load_vectorstore()

# Create retriever and pipeline
retriever = HybridRetriever(vectorstore, documents, config)
pipeline = RAGPipeline(retriever, config)

# Query
response = pipeline.query("What causes wormhole defects in FSW?")
print(response.answer)

for source in response.sources:
    print(f"- {source.doc_name} (Page {source.page_num})")
```

## Running Tests

```bash
pytest tests/ -v --cov=rag_demo
```

## Pre-commit Checks

```bash
./check-quality.sh
```
