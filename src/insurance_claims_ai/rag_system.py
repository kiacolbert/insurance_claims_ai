"""RAG system with proper dependency injection for testing"""

import chromadb
from pathlib import Path
import os
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass


class LLMClient(Protocol):
    """Protocol for LLM clients (enables mocking)"""
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate response from prompt"""
        ...


@dataclass
class RAGResponse:
    """Structured RAG response"""
    answer: str
    sources: List[Dict]
    context: List[str]
    cached: bool = False


class AnthropicClient:
    """Production Anthropic client"""
    
    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response using Anthropic API"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Return deterministic mock response"""
        # Extract question from prompt for context-aware mocking
        if "deductible" in prompt.lower():
            return "Your collision deductible is $500 per incident."
        elif "claim" in prompt.lower():
            return "To file a claim, call 1-800-CLAIMS within 24 hours."
        else:
            return "Based on the policy information provided, here is the answer."


class RAGSystem:
    """RAG system with dependency injection"""
    
    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        collection_name: str = "insurance_docs"
    ):
        """
        Initialize RAG system
        
        Args:
            chroma_path: Path to ChromaDB (defaults to env or standard location)
            llm_client: LLM client (injected for testing, auto-created for production)
            collection_name: Name of the collection
        """
        # Initialize ChromaDB
        if chroma_path is None:
            chroma_path = self._get_default_chroma_path()
        
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self._get_or_create_collection(collection_name)
        
        # Initialize LLM client
        if llm_client is None:
            # Production: create real client
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required")
            self.llm_client = AnthropicClient(api_key)
        else:
            # Testing: use injected mock
            self.llm_client = llm_client
    
    @staticmethod
    def _get_default_chroma_path() -> Path:
        """Get default ChromaDB path"""
        if env_path := os.getenv("CHROMA_DB_PATH"):
            return Path(env_path)
        
        # Default to project root / chroma_db
        project_root = Path(__file__).parent.parent.parent
        return project_root / "chroma_db"
    
    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection"""
        try:
            return self.client.get_collection(name=name)
        except Exception:
            return self.client.create_collection(name=name)
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        use_cache: bool = True
    ) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            use_cache: Whether to use caching (handled at API layer)
        
        Returns:
            RAGResponse with answer, sources, and context
        """
        # 1. Retrieve relevant documents from vector store
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        
        # 2. Extract documents and metadata
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                context=[],
                cached=False
            )
        
        # 3. Build context for LLM
        context = "\n\n".join(documents)
        
        # 4. Create prompt
        prompt = self._build_prompt(question, context)
        
        # 5. Generate answer using LLM
        answer = self.llm_client.generate(prompt)
        
        # 6. Return structured response
        return RAGResponse(
            answer=answer,
            sources=metadatas,
            context=documents,
            cached=False
        )
    
    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Based on the following insurance policy information, answer the question.

Policy Information:
{context}

Question: {question}

Answer concisely and only use information from the policy documents provided."""


# Singleton pattern for application use
_rag_system: Optional[RAGSystem] = None


def get_rag_system(
    chroma_path: Optional[Path] = None,
    llm_client: Optional[LLMClient] = None
) -> RAGSystem:
    """
    Get or create RAG system instance (singleton pattern)
    
    Args:
        chroma_path: Override default ChromaDB path
        llm_client: Override default LLM client (for testing)
    """
    global _rag_system
    
    if _rag_system is None:
        _rag_system = RAGSystem(
            chroma_path=chroma_path,
            llm_client=llm_client
        )
    
    return _rag_system


def reset_rag_system():
    """Reset singleton (useful for testing)"""
    global _rag_system
    _rag_system = None