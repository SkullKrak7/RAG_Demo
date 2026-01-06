# FSW RAG Demo: Friction Stir Welding Defect Analysis System

A Retrieval-Augmented Generation (RAG) system for intelligent defect analysis in Friction Stir Welding (FSW) processes, combining document retrieval with real-time sensor data analysis.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-demo-skullkrak7.streamlit.app/)

🌐 Access the live app: [https://rag-demo-skullkrak7.streamlit.app/](https://rag-demo-skullkrak7.streamlit.app/)

## Overview

FSW RAG Demo demonstrates three progressive implementations of RAG systems for FSW defect analysis:

1. **RAG 1: Simple QA** - Basic document question-answering
2. **RAG 2: Multi-Doc Filtering** - Metadata-driven document filtering
3. **RAG 3: Sensor Fusion** - Advanced sensor data + document integration

## Features

- **Interactive Streamlit UI** with three demo modes
- **Pre-built Vector Store** for instant loading (< 2 seconds)
- **Dual LLM Support**: Ollama (development) and HuggingFace (production)
- **Real-time Sensor Analysis** with threshold monitoring
- **ISO 25239 Compliant** prompts for FSW standards
- **Metadata-driven Retrieval** for targeted information access

## Architecture

### System Components

- **Frontend**: Streamlit web application
- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: sentence-transformers/paraphrase-MiniLM-L3-v2
- **LLM**: meta-llama/Llama-3.1-8B-Instruct (via HuggingFace API)
- **Document Processing**: PyPDF + RecursiveCharacterTextSplitter

### File Structure

```
RAG_Demo/
├── app.py                    # Main Streamlit application (268 lines)
├── build_vectorstore.py      # Vector store builder (49 lines)
├── rag_basic.py             # Demo 1: Basic RAG (104 lines)
├── rag_multidoc.py          # Demo 2: Multi-document filtering (157 lines)
├── rag_sensorfusion.py      # Demo 3: Sensor fusion (209 lines)
├── requirements.txt         # Python dependencies
├── .devcontainer/           # VS Code DevContainer configuration
│   └── devcontainer.json
├── data/                    # FSW documents and sensor data
│   ├── fsw_doc1.pdf        # ISO 25239-5
│   ├── fsw_doc2.pdf        # FSW-Tech Handbook
│   ├── fsw_doc3.pdf        # Defects Review
│   ├── fsw_doc4.pdf        # PhD Thesis
│   ├── fsw_doc5.pdf        # NDE Paper
│   └── sensor_log.csv      # Sensor readings with defect events
├── vectorstore/             # Pre-built ChromaDB (generated)
└── chroma_db/              # Alternative ChromaDB storage (generated)
```

## Installation

### Prerequisites

- Python 3.11 or higher
- HuggingFace API token (for production app)
- Ollama installed locally (for development demos)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SkullKrak7/RAG_Demo.git
   cd RAG_Demo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build vector store** (one-time setup):
   ```bash
   python build_vectorstore.py
   ```
   This generates embeddings from the 5 FSW PDF documents and stores them in `./vectorstore`.

4. **Set up HuggingFace API token**:
   ```bash
   export HF_TOKEN="your_huggingface_api_token_here"
   ```
   Or add it to Streamlit secrets for cloud deployment:
   ```toml
   # .streamlit/secrets.toml
   HF_TOKEN = "your_token_here"
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   The app will be available at `http://localhost:8501`

## Usage

### Production Application (app.py)

The main Streamlit app provides three interactive modes:

#### RAG 1: Simple QA
- Enter questions about FSW processes
- System retrieves relevant document chunks (k=5)
- Returns ISO-compliant answers with source attribution
- Example: "What causes wormhole defects in FSW?"

#### RAG 2: Multi-Doc Filtering
- Filter by document type: ISO Standards, Manuals, or Procedures
- Targeted retrieval based on metadata
- Shows which document types contributed to the answer
- Example: Query "welding parameters" filtered to ISO Standards only

#### RAG 3: Sensor Fusion
- Displays defect events from sensor log
- Select a defect to analyze (wormhole, tunnel, surface_crack, excessive_flash)
- System identifies threshold violations:
  - RPM > 650
  - Force > 14.0 kN
  - Temperature > 500°C
- Combines sensor context with document retrieval
- Generates root cause analysis based on FSW standards

### Development Demos

Three standalone scripts demonstrate RAG concepts using local Ollama:

#### rag_basic.py
```bash
python rag_basic.py
```
- Simple document Q&A with command-line interface
- Uses Ollama llama3.2:1b model (local)
- Interactive loop: ask questions, type 'quit' to exit

#### rag_multidoc.py
```bash
python rag_multidoc.py
```
- Demonstrates metadata-based filtering
- Five filtering options: All, Standards, Manuals, Procedures, High Priority
- Shows document types used in each answer

#### rag_sensorfusion.py
```bash
python rag_sensorfusion.py
```
- Loads sensor data from CSV
- Lists all defect events
- Provides detailed root cause analysis combining sensor readings and documentation

## Configuration

### Sensor Thresholds

Edit thresholds in `app.py` and `rag_sensorfusion.py`:
```python
THRESHOLDS = {
    'rpm_max': 650,
    'force_max_kn': 14.0,
    'temp_max_c': 500
}
```

### Chunk Settings

Modify text splitting parameters in `build_vectorstore.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=50       # Overlap for context preservation
)
```

### LLM Models

**Production (app.py)**:
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Temperature: 0.05 (low for consistency)
- Max tokens: 512
- Provider: Novita (via HuggingFace)

**Development (demo scripts)**:
- Model: `llama3.2:1b` (via Ollama)
- Temperature: 0 (deterministic)
- Endpoint: `http://localhost:11434`

## Data

### Sensor Log Format

The `data/sensor_log.csv` file contains:
- **timestamp**: Date-time of measurement (YYYY-MM-DD HH:MM:SS)
- **rpm**: Rotational speed (revolutions per minute)
- **force_kn**: Applied force (kilonewtons)
- **temperature_c**: Temperature (Celsius)
- **defect_type**: Detected defect or 'none'

### Document Corpus

Five FSW technical documents covering:
- ISO 25239-5 international standard
- Technical handbooks and procedures
- Research papers on defect analysis
- Non-destructive evaluation methods

## Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `HF_TOKEN` to secrets
4. Deploy

### Local Development

For development with local Ollama:

1. Install [Ollama](https://ollama.ai/)
2. Pull the model:
   ```bash
   ollama pull llama3.2:1b
   ```
3. Run development demos:
   ```bash
   python rag_basic.py
   python rag_multidoc.py
   python rag_sensorfusion.py
   ```

## Project Structure Details

### app.py - Main Application
- **Lines**: 268
- **Functions**:
  - `load_vectorstore()`: Loads pre-built ChromaDB (cached)
  - `get_llm()`: Initializes HuggingFace LLM
  - `setup_qa_chain()`: Creates RetrievalQA chain with custom prompts
- **UI Components**: Sidebar navigation, three demo modes, expandable source views

### build_vectorstore.py - Vector Store Builder
- **Lines**: 49
- **Process**:
  1. Loads 5 PDF documents from `data/` directory
  2. Splits into 500-character chunks with 50-character overlap
  3. Generates embeddings using MiniLM-L3-v2
  4. Persists to ChromaDB in `./vectorstore`
- **Execution**: Run once after document updates

### Demo Scripts
- **rag_basic.py** (104 lines): Simple Q&A with local Ollama
- **rag_multidoc.py** (157 lines): Metadata filtering demonstration
- **rag_sensorfusion.py** (209 lines): Sensor data + document fusion

## Technical Specifications

### Embedding Model
- **Name**: sentence-transformers/paraphrase-MiniLM-L3-v2
- **Dimensions**: 384
- **Size**: ~50MB
- **Speed**: Fast inference (20-50ms per query)

### Vector Store
- **Database**: ChromaDB
- **Persistence**: Disk-based (./vectorstore)
- **Similarity Metric**: Cosine similarity
- **Index Size**: ~500MB (for 5 documents)

### LLM Configuration
- **Production Model**: meta-llama/Llama-3.1-8B-Instruct
- **Context Window**: 8192 tokens
- **Output Tokens**: 512 max
- **Temperature**: 0.05 (minimal creativity)
- **API Provider**: HuggingFace Inference (Novita)

## License

MIT License

Copyright (c) 2026 Sai Karthik Kagolanu
