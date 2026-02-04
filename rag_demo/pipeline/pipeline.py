"""RAG pipeline orchestrator."""

from typing import Iterator, Optional, List
from langchain_core.documents import Document

from rag_demo.core.config import RAGConfig
from rag_demo.core.models import RAGResponse
from rag_demo.core.exceptions import RetrievalError, GenerationError
from rag_demo.retrieval.retriever import HybridRetriever
from rag_demo.retrieval.reranker import Reranker
from rag_demo.generation.generator import LLMGenerator
from rag_demo.generation.formatter import ResponseFormatter
from rag_demo.observability.tracer import RAGTracer


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(
        self, retriever: HybridRetriever, config: RAGConfig, tracer: Optional[RAGTracer] = None
    ):
        self.retriever = retriever
        self.config = config
        self.reranker = Reranker(config)
        self.generator = LLMGenerator(config)
        self.tracer = tracer

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[{i}] {source}{page_info}:\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def query(self, question: str, stream: bool = False) -> RAGResponse:
        """Execute RAG query."""
        if self.tracer:
            return self._query_with_tracing(question, stream)
        return self._query_internal(question, stream)

    def _query_internal(self, question: str, stream: bool) -> RAGResponse:
        """Internal query execution."""
        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:
            raise RetrievalError("No documents retrieved")

        reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=self.config.rerank_top_k
        )

        context = self._format_context(reranked_docs)
        prompt = self.generator.create_prompt(question, context)

        if stream:
            answer_chunks = list(self.generator.stream(prompt))
            answer = "".join(answer_chunks)
        else:
            answer = self.generator.generate(prompt)

        sources = ResponseFormatter.format_sources(reranked_docs)

        return ResponseFormatter.create_response(
            query=question,
            answer=answer,
            sources=sources,
            retrieved_count=len(retrieved_docs),
            reranked_count=len(reranked_docs),
        )

    def _query_with_tracing(self, question: str, stream: bool) -> RAGResponse:
        """Query execution with Langfuse tracing."""

        @self.tracer.trace_query(name="rag_pipeline")
        def traced_query():
            retrieved_docs = self.retriever.retrieve(question)

            if not retrieved_docs:
                raise RetrievalError("No documents retrieved")

            self.tracer.log_retrieval(
                query=question, documents=retrieved_docs, retrieval_method="hybrid"
            )

            reranked_docs = self.reranker.rerank(
                question, retrieved_docs, top_k=self.config.rerank_top_k
            )

            context = self._format_context(reranked_docs)
            prompt = self.generator.create_prompt(question, context)

            if stream:
                answer_chunks = list(self.generator.stream(prompt))
                answer = "".join(answer_chunks)
            else:
                answer = self.generator.generate(prompt)

            self.tracer.log_generation(prompt=prompt, response=answer, model=self.config.model_name)

            sources = ResponseFormatter.format_sources(reranked_docs)

            return ResponseFormatter.create_response(
                query=question,
                answer=answer,
                sources=sources,
                retrieved_count=len(retrieved_docs),
                reranked_count=len(reranked_docs),
            )

        return traced_query()

    def stream_query(self, question: str) -> Iterator[str]:
        """Stream RAG query response."""
        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:
            raise RetrievalError("No documents retrieved")

        reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=self.config.rerank_top_k
        )

        context = self._format_context(reranked_docs)
        prompt = self.generator.create_prompt(question, context)

        for chunk in self.generator.stream(prompt):
            yield chunk
