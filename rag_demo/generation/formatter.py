"""Response formatter with source attribution."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from rag_demo.core.models import Source, RAGResponse


class ResponseFormatter:
    """Format RAG responses with source citations."""
    
    @staticmethod
    def format_sources(documents: List[Document]) -> List[Source]:
        """Extract source citations from retrieved documents."""
        sources = []
        
        for doc in documents:
            source = Source(
                doc_name=doc.metadata.get("source", "Unknown"),
                page_num=doc.metadata.get("page"),
                chunk_text=doc.page_content,
                relevance_score=doc.metadata.get("score", 0.0),
                metadata=doc.metadata
            )
            sources.append(source)
        
        return sources
    
    @staticmethod
    def create_response(
        answer: str,
        documents: List[Document],
        latency_ms: float,
        confidence: float = 1.0,
        faithfulness_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """Create complete RAG response with citations."""
        sources = ResponseFormatter.format_sources(documents)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            faithfulness_score=faithfulness_score,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
    
    @staticmethod
    def format_citation(source: Source, index: int) -> str:
        """Format single source citation for display."""
        page_info = f", Page {source.page_num}" if source.page_num else ""
        return f"[{index}] {source.doc_name}{page_info} (Score: {source.relevance_score:.3f})"
