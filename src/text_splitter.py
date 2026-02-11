"""
Text chunking module for splitting documents into smaller pieces.

This is essential for RAG systems because:
1. LLMs have context limits
2. Smaller chunks = more precise retrieval
3. Better semantic matching
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """Handles splitting documents into chunks for embedding."""
    
    def __init__(
        self, 
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
                          (helps maintain context across boundaries)
        
        Overlap Example:
        Chunk 1: "...the model learns patterns from data."
        Chunk 2: "patterns from data. The algorithm then..."
                  ^^^^^^^^^^^^^^^^ (overlap preserves context)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Paragraphs (\n\n)
        # 2. Sentences (. ! ?)
        # 3. Words (spaces)
        # 4. Characters (as last resort)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text string to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
            
        Time Complexity: O(n) where n = length of text
        Space Complexity: O(n) to store chunks
        """
        # Split the text
        chunks = self.splitter.split_text(text)
        
        # Format chunks with metadata
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['chunk_size'] = len(chunk)
            
            formatted_chunks.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        return formatted_chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            Flattened list of all chunks from all documents
            
        Example:
            Input: [
                {'content': 'doc1 text...', 'metadata': {'source': 'file1.txt'}},
                {'content': 'doc2 text...', 'metadata': {'source': 'file2.txt'}}
            ]
            Output: [
                {'content': 'doc1 chunk1...', 'metadata': {'source': 'file1.txt', 'chunk_index': 0}},
                {'content': 'doc1 chunk2...', 'metadata': {'source': 'file1.txt', 'chunk_index': 1}},
                {'content': 'doc2 chunk1...', 'metadata': {'source': 'file2.txt', 'chunk_index': 0}},
                ...
            ]
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Chunk this document
            doc_chunks = self.chunk_text(content, metadata)
            all_chunks.extend(doc_chunks)
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks.
        
        Useful for debugging and optimization.
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }


# Test function
def test_chunker():
    """Test the text chunker."""
    
    # Sample document
    test_doc = """
Machine Learning Overview

Machine learning is a subset of artificial intelligence. It enables systems to learn from data.

There are three main types of machine learning:

Supervised Learning: Uses labeled data to train models. The algorithm learns to map inputs to outputs based on example pairs.

Unsupervised Learning: Finds patterns in unlabeled data. Common techniques include clustering and dimensionality reduction.

Reinforcement Learning: Learns through trial and error. The agent receives rewards or penalties based on its actions.

Deep learning is a specialized form of machine learning that uses neural networks with multiple layers.
    """.strip()
    
    # Create chunker
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    
    # Test single document
    print("🧪 Testing Text Chunker\n")
    print(f"Original text length: {len(test_doc)} characters\n")
    
    chunks = chunker.chunk_text(
        test_doc, 
        metadata={'source': 'test.txt', 'topic': 'ML'}
    )
    
    print(f"✓ Created {len(chunks)} chunks\n")
    
    # Display chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Size: {chunk['metadata']['chunk_size']} chars")
        print(f"  Content: {chunk['content'][:80]}...")
        print()
    
    # Test multiple documents
    print("Testing with multiple documents...")
    test_docs = [
        {
            'content': test_doc,
            'metadata': {'source': 'ml_basics.txt', 'category': 'education'}
        },
        {
            'content': "Python is a programming language. It's great for ML.",
            'metadata': {'source': 'python.txt', 'category': 'programming'}
        }
    ]
    
    all_chunks = chunker.chunk_documents(test_docs)
    print(f"✓ Created {len(all_chunks)} total chunks from {len(test_docs)} documents\n")
    
    # Get statistics
    stats = chunker.get_chunk_stats(all_chunks)
    print("📊 Chunk Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Chunker test passed!")


if __name__ == "__main__":
    test_chunker()