"""Langfuse tracer for RAG observability."""

from typing import Optional, Dict, Any, List
from functools import wraps
import time
from langfuse import Langfuse, observe

from rag_demo.core.config import RAGConfig
from rag_demo.core.exceptions import RAGException


class RAGTracer:
    """Observability tracer for RAG pipeline using Langfuse."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.enabled = config.langfuse_enabled
        self.client: Optional[Langfuse] = None
        
        if self.enabled:
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                raise RAGException("Langfuse keys required when langfuse_enabled=True")
            
            self.client = Langfuse(
                public_key=config.langfuse_public_key.get_secret_value(),
                secret_key=config.langfuse_secret_key.get_secret_value(),
                host="https://cloud.langfuse.com"
            )
    
    def trace_query(self, name: str = "rag_query"):
        """Decorator to trace RAG query execution."""
        def decorator(func):
            if not self.enabled:
                return func
            
            @observe(name=name)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def log_retrieval(self, query: str, num_chunks: int, latency_ms: float):
        """Log retrieval metrics."""
        if not self.enabled or not self.client:
            return
        
        self.client.span(
            name="retrieval",
            input=query,
            output={"num_chunks": num_chunks},
            metadata={"latency_ms": latency_ms}
        )
    
    def log_generation(
        self,
        prompt: str,
        answer: str,
        tokens_used: int,
        latency_ms: float
    ):
        """Log generation metrics."""
        if not self.enabled or not self.client:
            return
        
        self.client.generation(
            name="generation",
            input=prompt,
            output=answer,
            usage={"total_tokens": tokens_used},
            metadata={"latency_ms": latency_ms}
        )
    
    def score_feedback(self, trace_id: str, score: float, comment: Optional[str] = None):
        """Log user feedback score."""
        if not self.enabled or not self.client:
            return
        
        self.client.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment
        )
    
    def flush(self):
        """Flush pending traces."""
        if self.enabled and self.client:
            self.client.flush()
