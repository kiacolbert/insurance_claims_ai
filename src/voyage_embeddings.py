"""
LangChain wrapper for Voyage AI embeddings.

This allows us to use Voyage AI with LangChain's vector stores.
"""

from langchain_core.embeddings import Embeddings
from typing import List
import voyageai
import os


class VoyageEmbeddings(Embeddings):
    """LangChain-compatible Voyage AI embeddings."""
    
    def __init__(self, model: str = "voyage-4"):
        """Initialize Voyage embeddings."""
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        return result.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed search query."""
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="query"
        )
        return result.embeddings[0]