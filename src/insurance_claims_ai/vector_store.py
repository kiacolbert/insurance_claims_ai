"""
Vector store module using Voyage AI embeddings + ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from voyage_embeddings import VoyageEmbeddings
import os


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB + Voyage AI."""
    
    def __init__(
        self, 
        collection_name: str = "qa_documents",
        persist_directory: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        embedding_model: str = "voyage-4"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name for the document collection
            persist_directory: Where to store the database locally
            embedding_model: Voyage AI model (voyage-4, voyage-large-4, voyage-code-4)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Voyage embeddings
        self.embeddings = VoyageEmbeddings(model=embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize vector store
        self.vector_store = None
        
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dicts for each chunk
            ids: Optional list of unique IDs for each chunk
            
        Returns:
            List of document IDs that were added
            
        Time Complexity: O(n * t * d) where:
            n = number of documents
            t = avg tokens per document  
            d = embedding dimensions (1024 for voyage-4)
        """
        # Create vector store if it doesn't exist
        if self.vector_store is None:
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            # Add to existing store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        # Return IDs (generate if not provided)
        if ids:
            return ids
        else:
            return [f"doc_{i}" for i in range(len(texts))]
    
    def load_existing(self):
        """
        Load an existing vector store from disk.
        
        Useful when restarting the application.
        """
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4
    ) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
            
        Time Complexity: O(log n * d) with HNSW index (Chroma default)
        """
        if self.vector_store is None:
            self.load_existing()
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return formatted_results
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4
    ) -> List[tuple]:
        """
        Search with relevance scores.
        
        Returns:
            List of (document, score) tuples
            Lower scores = more similar
        """
        if self.vector_store is None:
            self.load_existing()
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def delete_collection(self):
        """Delete the entire collection (useful for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"⚠ Could not delete collection: {e}")
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection."""
        if self.vector_store is None:
            self.load_existing()
        return self.vector_store._collection.count()


# Test function
def test_vector_store():
    """Test the vector store with Voyage AI embeddings."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create test documents
    test_docs = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "The weather forecast predicts sunny skies and warm temperatures today.",
        "Deep learning neural networks are a powerful subset of machine learning."
    ]
    
    test_metadata = [
        {"source": "python_docs", "topic": "programming"},
        {"source": "ml_basics", "topic": "machine_learning"},
        {"source": "weather_api", "topic": "weather"},
        {"source": "dl_guide", "topic": "deep_learning"}
    ]
    
    # Initialize store
    print("🧪 Testing Vector Store with Voyage AI\n")
    store = VectorStore(
        collection_name="test_collection",
        embedding_model="voyage-4"
    )
    
    # Clean up any existing test collection
    store.delete_collection()
    
    # Add documents
    print("📝 Adding documents...")
    ids = store.add_documents(test_docs, test_metadata)
    print(f"✓ Added {len(ids)} documents")
    print(f"✓ Collection now has {store.get_collection_count()} documents\n")
    
    # Test search
    print("🔍 Testing semantic search...")
    query = "How do ML algorithms work?"
    results = store.similarity_search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {result['content']}")
        print(f"  Topic: {result['metadata']['topic']}")
        print()
    
    # Test with scores
    print("📊 Testing search with similarity scores...")
    results_with_scores = store.similarity_search_with_score(query, k=3)
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"Result {i} (distance: {score:.4f}):")
        print(f"  {doc.page_content[:60]}...")
        print()
    
    # Clean up
    store.delete_collection()
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_vector_store()