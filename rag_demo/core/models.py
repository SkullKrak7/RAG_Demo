"""Data models for RAG responses."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Source:
    """Source document citation."""
    
    doc_name: str
    page_num: Optional[int]
    chunk_text: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    
    answer: str
    sources: List[Source]
    confidence: float
    faithfulness_score: Optional[float]
    latency_ms: float
    metadata: Dict[str, Any]
