"""Streamlit UI for FSW RAG system."""

import streamlit as st
from pathlib import Path

from rag_demo.core.config import RAGConfig
from rag_demo.core.exceptions import RetrievalError, GenerationError
from rag_demo.ingestion.builder import VectorStoreBuilder
from rag_demo.retrieval.retriever import HybridRetriever
from rag_demo.pipeline.pipeline import RAGPipeline
from rag_demo.observability.tracer import RAGTracer


st.set_page_config(
    page_title="FSW Defect Analysis RAG",
    page_icon="🔧",
    layout="wide"
)


@st.cache_resource
def load_config():
    """Load configuration."""
    return RAGConfig()


@st.cache_resource
def initialize_system(_config):
    """Initialize RAG system components."""
    builder = VectorStoreBuilder(_config)
    
    vectorstore = builder.load_vectorstore()
    
    documents = []
    for doc_id in vectorstore.get()["ids"]:
        doc = vectorstore.get(ids=[doc_id])
        documents.append(doc)
    
    retriever = HybridRetriever(vectorstore, documents, _config)
    
    tracer = None
    if _config.langfuse_enabled:
        tracer = RAGTracer(_config)
    
    pipeline = RAGPipeline(retriever, _config, tracer=tracer)
    
    return pipeline, tracer


def render_source(source, index):
    """Render single source citation."""
    with st.expander(f"Source {index}: {source.doc_name}"):
        if source.page_num:
            st.caption(f"Page {source.page_num}")
        st.caption(f"Relevance Score: {source.relevance_score:.3f}")
        st.text(source.chunk_text)


def main():
    """Main Streamlit application."""
    st.title("🔧 FSW Defect Analysis RAG System")
    st.markdown("Ask questions about friction stir welding defects, root causes, and corrective actions.")
    
    try:
        config = load_config()
        pipeline, tracer = initialize_system(config)
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Make sure vector store exists. Run document ingestion first.")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    render_source(source, i)
    
    if prompt := st.chat_input("Ask about FSW defects..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Retrieving relevant documents..."):
                    response = pipeline.query(prompt, stream=False)
                
                message_placeholder.markdown(response.answer)
                
                st.markdown("**Sources:**")
                for i, source in enumerate(response.sources, 1):
                    render_source(source, i)
                
                st.caption(f"Retrieved: {response.metadata.get('retrieved_count', 0)} | "
                          f"Reranked: {response.metadata.get('reranked_count', 0)}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources
                })
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
                        if tracer:
                            tracer.score_feedback(1.0, "user_feedback")
                        st.success("Feedback recorded!")
                
                with col2:
                    if st.button("👎 Not Helpful", key=f"down_{len(st.session_state.messages)}"):
                        if tracer:
                            tracer.score_feedback(0.0, "user_feedback")
                        st.info("Feedback recorded!")
            
            except RetrievalError as e:
                st.error(f"Retrieval failed: {e}")
            except GenerationError as e:
                st.error(f"Generation failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    
    with st.sidebar:
        st.header("System Configuration")
        st.metric("Model", config.model_name.split("/")[-1])
        st.metric("Temperature", config.temperature)
        st.metric("Retrieval K", config.retrieval_k)
        st.metric("Rerank Top K", config.rerank_top_k)
        
        if config.langfuse_enabled:
            st.success("Langfuse Tracing: Enabled")
        else:
            st.info("Langfuse Tracing: Disabled")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
