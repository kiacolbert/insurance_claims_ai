"""
Main QA pipeline using Anthropic ecosystem:
- Voyage AI for embeddings
- ChromaDB for vector storage  
- Claude for Q&A (coming tomorrow!)
"""

from dotenv import load_dotenv
# from document_loader import DocumentLoader
from text_splitter import TextChunker
from vector_store import VectorStore
import json
import os
import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Encoding to use (cl100k_base is used by most modern models)
        
    Returns:
        Number of tokens
        
    Note: Voyage AI uses similar tokenization to OpenAI models,
          so cl100k_base is a good approximation.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def estimate_embedding_cost(total_tokens: int, model: str = "voyage-2") -> dict:
    """
    Estimate cost for embedding generation.
    
    Voyage AI Pricing (as of 2024):
    - voyage-2: $0.10 per 1M tokens
    - voyage-large-2: $0.12 per 1M tokens
    - voyage-code-2: $0.10 per 1M tokens
    
    Free tier: 100M tokens/month
    """
    pricing = {
        "voyage-2": 0.10,
        "voyage-large-2": 0.12,
        "voyage-code-2": 0.10
    }
    
    cost_per_million = pricing.get(model, 0.10)
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    
    return {
        "total_tokens": total_tokens,
        "cost_per_million": cost_per_million,
        "estimated_cost_usd": estimated_cost,
        "within_free_tier": total_tokens <= 100_000_000  # 100M free
    }


def analyze_token_usage(chunks: list) -> dict:
    """
    Analyze token usage across all chunks.
    
    Returns detailed statistics about token distribution.
    """
    token_counts = []
    total_tokens = 0
    
    for chunk in chunks:
        tokens = count_tokens(chunk['content'])
        token_counts.append(tokens)
        total_tokens += tokens
    
    if not token_counts:
        return {
            "total_tokens": 0,
            "avg_tokens_per_chunk": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "num_chunks": 0
        }
    
    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "num_chunks": len(token_counts),
        "token_distribution": {
            "0-100": sum(1 for t in token_counts if t <= 100),
            "101-200": sum(1 for t in token_counts if 100 < t <= 200),
            "201-300": sum(1 for t in token_counts if 200 < t <= 300),
            "301-400": sum(1 for t in token_counts if 300 < t <= 400),
            "400+": sum(1 for t in token_counts if t > 400),
        }
    }


def main():
    """Run the complete pipeline: load → chunk → embed → store → search."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.getenv("VOYAGE_API_KEY"):
        raise ValueError("VOYAGE_API_KEY not found in .env file")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠ Warning: ANTHROPIC_API_KEY not found (needed for Claude tomorrow)")
    
    print("=" * 70)
    print("🚀 QA Pipeline with Token Analysis (Anthropic Stack)")
    print("=" * 70)
    print()
    
    # Step 1: Load documents
    print("📄 Step 1: Loading documents...")
    # loader = DocumentLoader("./data/documents")
    # documents = loader.load_all()
    with open ('insurance_docs.json') as f:
        documents = json.load(f)
    print(f"   ✓ Loaded {len(documents)} documents\n")
    
    # Step 2: Chunk documents
    print("✂️  Step 2: Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    # Get statistics
    stats = chunker.get_chunk_stats(chunks)
    print(f"   ✓ Created {stats['total_chunks']} chunks")
    print(f"   ✓ Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
    print(f"   ✓ Min/Max chunk size: {stats['min_chunk_size']}/{stats['max_chunk_size']} chars\n")
    
    # Step 2.5: Analyze token usage
    print("🔢 Step 2.5: Analyzing token usage...")
    token_stats = analyze_token_usage(chunks)
    
    print(f"   ✓ Total tokens to embed: {token_stats['total_tokens']:,}")
    print(f"   ✓ Avg tokens per chunk: {token_stats['avg_tokens_per_chunk']:.1f}")
    print(f"   ✓ Min/Max tokens: {token_stats['min_tokens']}/{token_stats['max_tokens']}")
    
    print(f"\n   📊 Token Distribution:")
    for range_label, count in token_stats['token_distribution'].items():
        if count > 0:
            percentage = (count / token_stats['num_chunks']) * 100
            print(f"      {range_label:>10} tokens: {count:3d} chunks ({percentage:5.1f}%)")
    
    # Estimate cost
    cost_info = estimate_embedding_cost(
        token_stats['total_tokens'], 
        model="voyage-2"
    )
    
    print(f"\n   💰 Cost Estimate (Voyage-2):")
    print(f"      Total tokens: {cost_info['total_tokens']:,}")
    print(f"      Cost: ${cost_info['estimated_cost_usd']:.6f}")
    if cost_info['within_free_tier']:
        print(f"      ✓ Within free tier (100M tokens/month)")
    else:
        print(f"      ⚠ Exceeds free tier!")
    print()
    
    # Step 3: Create embeddings and store
    print("🧠 Step 3: Generating embeddings (Voyage AI) and storing...")
    store = VectorStore(
        collection_name="qa_docs",
        embedding_model="voyage-2"
    )
    
    # Extract text and metadata from chunks
    texts = [chunk['content'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Add to vector store
    doc_ids = store.add_documents(texts, metadatas)
    print(f"   ✓ Stored {len(doc_ids)} chunks with Voyage embeddings")
    print(f"   ✓ Total in collection: {store.get_collection_count()}")
    
    # Calculate storage size
    embedding_dims = 1024  # voyage-2 uses 1024 dimensions
    bytes_per_float = 4
    total_embedding_size = len(doc_ids) * embedding_dims * bytes_per_float
    
    print(f"   ✓ Embedding storage: ~{total_embedding_size / 1024:.1f} KB")
    print()
    
    # Step 4: Test retrieval
    print("🔍 Step 4: Testing semantic search...\n")
    
    test_queries = [
        "What is machine learning?",
        "How does Python work?",
        "Tell me about data processing"
    ]
    
    # Track query tokens
    total_query_tokens = 0
    
    for query in test_queries:
        query_tokens = count_tokens(query)
        total_query_tokens += query_tokens
        
        print(f"Query: '{query}' ({query_tokens} tokens)")
        results = store.similarity_search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  Result {i}:")
            content_preview = result['content'][:80].replace('\n', ' ')
            print(f"    {content_preview}...")
            print(f"    Source: {result['metadata'].get('source', 'unknown')}")
        print()
    
    # Final summary
    print("=" * 70)
    print("📈 FINAL TOKEN SUMMARY")
    print("=" * 70)
    print(f"Documents processed:       {len(documents)}")
    print(f"Chunks created:            {len(chunks)}")
    print(f"Total embedding tokens:    {token_stats['total_tokens']:,}")
    print(f"Total query tokens:        {total_query_tokens}")
    print(f"Total tokens used:         {token_stats['total_tokens'] + total_query_tokens:,}")
    print(f"Estimated cost:            ${cost_info['estimated_cost_usd']:.6f}")
    print(f"Storage size (embeddings): ~{total_embedding_size / 1024:.1f} KB")
    print("=" * 70)
    print()
    
    print("✅ Pipeline complete! Ready for Claude integration tomorrow.")
    
    return store


if __name__ == "__main__":
    store = main()