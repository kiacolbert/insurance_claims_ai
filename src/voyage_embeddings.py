"""
LangChain wrapper for Voyage AI embeddings with rate limiting.
"""

from langchain_core.embeddings import Embeddings
from typing import List
from embeddings import EmbeddingGenerator


class VoyageEmbeddings(Embeddings):
    """LangChain-compatible Voyage AI embeddings with automatic rate limiting."""
    
    def __init__(self, model: str = "voyage-2"):
        """Initialize Voyage embeddings with rate limiting."""
        self.generator = EmbeddingGenerator(model=model)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search documents with automatic batching and rate limiting.
        
        This will handle large lists automatically!
        """
        return self.generator.embed_texts(texts, show_progress=True)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed search query."""
        return self.generator.embed_query(text)