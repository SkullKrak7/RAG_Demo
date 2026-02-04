"""LLM generator with streaming support."""

from typing import Iterator, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

from rag_demo.core.config import RAGConfig
from rag_demo.core.exceptions import GenerationError


class LLMGenerator:
    """LLM generation with streaming support."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._llm = None
        self._chat_model = None
    
    def _get_llm(self) -> HuggingFaceEndpoint:
        """Initialize LLM endpoint."""
        if self._llm is None:
            try:
                self._llm = HuggingFaceEndpoint(
                    repo_id=self.config.model_name,
                    task="text-generation",
                    huggingfacehub_api_token=self.config.hf_token.get_secret_value(),
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                raise GenerationError(f"Failed to initialize LLM: {e}")
        
        return self._llm
    
    def _get_chat_model(self) -> ChatHuggingFace:
        """Initialize chat model wrapper."""
        if self._chat_model is None:
            llm = self._get_llm()
            self._chat_model = ChatHuggingFace(llm=llm)
        
        return self._chat_model
    
    def generate(self, prompt: str) -> str:
        """Generate response without streaming."""
        try:
            llm = self._get_llm()
            return llm.invoke(prompt)
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Generate response with streaming."""
        try:
            llm = self._get_llm()
            for chunk in llm.stream(prompt):
                yield chunk
        except Exception as e:
            raise GenerationError(f"Streaming failed: {e}")
    
    def create_prompt(
        self,
        query: str,
        context: str,
        template: Optional[str] = None
    ) -> str:
        """Create formatted prompt from template."""
        if template is None:
            template = """You are an expert in friction stir welding. Use the following context from ISO 25239 standards and operational procedures to answer the question. Be specific about root causes and corrective actions.

Context: {context}

Question: {question}

Detailed Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return prompt.format(context=context, question=query)
