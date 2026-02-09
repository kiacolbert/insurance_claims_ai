"""
Embedding generation module using Voyage AI.

Voyage AI is Anthropic's recommended embeddings provider.
They offer state-of-the-art embeddings optimized for RAG.
"""

import voyageai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using Voyage AI."""
    
    def __init__(self, model: str = "voyage-4"):
        """
        Initialize embedding generator.
        
        Args:
            model: Voyage AI model to use
                   Options:
                   - voyage-4: General purpose, 1024 dims (recommended)
        
        Voyage Models Overview:
        - voyage-4: Best balance of speed/quality for most use cases
        """
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
            
        Time Complexity: O(n * t) where n = num texts, t = avg tokens/text
        Space Complexity: O(n * d) where d = embedding dimensions (1024 or 1536)
        """
        # Voyage AI can handle batches efficiently
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"  # Optimize for document storage
        )
        return result.embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Uses input_type="query" to optimize the embedding for search.
        This is important! Query embeddings should be optimized differently
        than document embeddings.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query"  # Optimize for search queries
        )
        return result.embeddings[0]


# Quick test function
def test_embeddings():
    """Test the embedding generator."""
    generator = EmbeddingGenerator()
    
    # Test batch embedding
    test_texts = [
        "Python is a programming language.",
        "Machine learning uses algorithms to learn from data.",
        "The weather today is sunny and warm."
    ]
    
    print("🧪 Testing Voyage AI Embeddings\n")
    print(f"Using model: {generator.model}")
    
    embeddings = generator.embed_texts(test_texts)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Each embedding has {len(embeddings[0])} dimensions")
    print(f"✓ First 5 values of embedding 1: {embeddings[0][:5]}")
    
    # Test query embedding
    query = "What is machine learning?"
    query_embedding = generator.embed_query(query)
    print(f"\n✓ Query embedding dimensions: {len(query_embedding)}")
    print(f"✓ First 5 values: {query_embedding[:5]}")
    
    # Test similarity (cosine similarity)
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    print("\n📊 Similarity scores (query vs documents):")
    for i, doc_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        print(f"   Doc {i+1}: {similarity:.4f} - {test_texts[i][:50]}...")
    
    print("\n✅ Embeddings test passed!")


if __name__ == "__main__":
    test_embeddings()