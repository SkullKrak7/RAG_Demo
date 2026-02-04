"""Test response formatter."""

import pytest
from langchain_core.documents import Document
from rag_demo.generation.formatter import ResponseFormatter
from rag_demo.core.models import Source, RAGResponse


def test_format_sources_extracts_metadata():
    """Format sources extracts all metadata correctly."""
    docs = [
        Document(
            page_content="Test content",
            metadata={"source": "doc1.pdf", "page": 5, "score": 0.95}
        )
    ]
    
    sources = ResponseFormatter.format_sources(docs)
    
    assert len(sources) == 1
    assert sources[0].doc_name == "doc1.pdf"
    assert sources[0].page_num == 5
    assert sources[0].relevance_score == 0.95
    assert sources[0].chunk_text == "Test content"


def test_format_sources_handles_missing_metadata():
    """Format sources handles missing metadata gracefully."""
    docs = [Document(page_content="Test", metadata={})]
    
    sources = ResponseFormatter.format_sources(docs)
    
    assert sources[0].doc_name == "Unknown"
    assert sources[0].page_num is None
    assert sources[0].relevance_score == 0.0


def test_create_response_builds_complete_response():
    """Create response builds complete RAG response."""
    docs = [
        Document(
            page_content="Content",
            metadata={"source": "test.pdf", "page": 1, "score": 0.9}
        )
    ]
    
    response = ResponseFormatter.create_response(
        answer="Test answer",
        documents=docs,
        latency_ms=150.0,
        confidence=0.85,
        faithfulness_score=0.92
    )
    
    assert response.answer == "Test answer"
    assert len(response.sources) == 1
    assert response.confidence == 0.85
    assert response.faithfulness_score == 0.92
    assert response.latency_ms == 150.0


def test_format_citation_with_page():
    """Format citation includes page number."""
    source = Source(
        doc_name="doc.pdf",
        page_num=10,
        chunk_text="text",
        relevance_score=0.88,
        metadata={}
    )
    
    citation = ResponseFormatter.format_citation(source, 1)
    
    assert "[1]" in citation
    assert "doc.pdf" in citation
    assert "Page 10" in citation
    assert "0.880" in citation


def test_format_citation_without_page():
    """Format citation works without page number."""
    source = Source(
        doc_name="doc.pdf",
        page_num=None,
        chunk_text="text",
        relevance_score=0.75,
        metadata={}
    )
    
    citation = ResponseFormatter.format_citation(source, 2)
    
    assert "[2]" in citation
    assert "doc.pdf" in citation
    assert "Page" not in citation
    assert "0.750" in citation
