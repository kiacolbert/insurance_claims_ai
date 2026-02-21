"""
Create Vector Database with Robust Rate Limiting
Handles Voyage AI rate limits: 3 RPM, 10K TPM
"""

import chromadb
from voyageai import Client as VoyageClient
from dotenv import load_dotenv
import os
import json
import time
import tiktoken
from tqdm import tqdm

load_dotenv()

# Rate limit configuration
MAX_TOKENS_PER_REQUEST = 2_000  # Conservative: 20% of 10K TPM limit
SECONDS_BETWEEN_REQUESTS = 30   # 2 requests per minute (under 3 RPM)

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def make_api_call_with_retry(client, texts, max_retries=3):
    """
    Make API call with exponential backoff retry.
    
    If we hit rate limit, wait and retry with increasing delays.
    """
    for attempt in range(max_retries):
        try:
            result = client.embed(
                texts,
                model="voyage-3",
                input_type="document"
            )
            return result.embeddings
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: 30s, 60s, 120s
                    wait_time = SECONDS_BETWEEN_REQUESTS * (2 ** attempt)
                    print(f"      ⚠️  Rate limit hit! Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ Rate limit error after {max_retries} retries")
                    raise
            else:
                # Some other error, don't retry
                print(f"      ❌ API Error: {error_msg}")
                raise
    
    raise Exception(f"Failed after {max_retries} retries")

def batch_texts_by_tokens(texts, max_tokens=MAX_TOKENS_PER_REQUEST):
    """
    Split texts into batches that respect token limits.
    Uses greedy bin packing algorithm.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        tokens = count_tokens(text)
        
        # If single text exceeds limit, include it anyway (will be its own batch)
        if tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([text])
            continue
        
        # Check if adding this text would exceed limit
        if current_tokens + tokens <= max_tokens:
            current_batch.append(text)
            current_tokens += tokens
        else:
            # Start new batch
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = tokens
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
    
    return batches

def main():
    print("=" * 70)
    print("CREATE VECTOR DATABASE (WITH RATE LIMITING)")
    print("=" * 70)
    
    # Initialize clients
    print("\n1. Initializing clients...")
    voyage_client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))
    CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("   ✅ Clients initialized")
    
    # Load documents
    print("\n2. Loading documents...")
    try:
        with open('insurance_docs.json') as f:
            docs = json.load(f)
        print(f"   ✅ Loaded {len(docs)} documents")
    except FileNotFoundError:
        print("   ❌ Error: insurance_docs.json not found!")
        return
    
    # Delete existing collection if it exists
    print("\n3. Setting up collection...")
    try:
        chroma_client.delete_collection("insurance_docs")
        print("   ✅ Deleted old collection")
    except:
        print("   ℹ️  No existing collection to delete")
    
    collection = chroma_client.create_collection(
        name="insurance_docs",
        metadata={"description": "Insurance policy Q&A"}
    )
    print("   ✅ Created new collection: 'insurance_docs'")
    
    # Prepare data
    print("\n4. Preparing documents...")
    texts = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(docs):
        content = doc.get('content', '')
        if not content:
            continue
            
        texts.append(content)
        
        question = doc.get('question', 'N/A')
        if len(question) > 100:
            question = question[:97] + "..."
            
        metadatas.append({
            'source': f"InsuranceQA Document {i+1}",
            'question': question,
            'doc_index': i
        })
        ids.append(f"doc_{i}")
    
    print(f"   ✅ Prepared {len(texts)} documents")
    
    # Count total tokens
    print("\n5. Analyzing token usage...")
    total_tokens = sum(count_tokens(t) for t in texts)
    print(f"   📊 Total tokens: {total_tokens:,}")
    
    # Create batches
    batches = batch_texts_by_tokens(texts, MAX_TOKENS_PER_REQUEST)
    print(f"   📦 Split into {len(batches)} batches")
    
    # Estimate time
    estimated_time = len(batches) * SECONDS_BETWEEN_REQUESTS
    print(f"   ⏱️  Estimated time: ~{estimated_time // 60}m {estimated_time % 60}s")
    print(f"   💡 This is slow to avoid rate limit errors!")
    
    # Generate embeddings with rate limiting
    print("\n6. Generating embeddings with rate limiting...")
    all_embeddings = []
    batch_start_indices = []
    current_index = 0
    
    for batch_num, batch in enumerate(tqdm(batches, desc="   Processing"), 1):
        batch_tokens = sum(count_tokens(t) for t in batch)
        
        # Wait before request (except first one)
        if batch_num > 1:
            print(f"\n   ⏳ Waiting {SECONDS_BETWEEN_REQUESTS}s for rate limit...")
            time.sleep(SECONDS_BETWEEN_REQUESTS)
        
        # Make the API call with retry logic
        print(f"\n   🔄 Batch {batch_num}/{len(batches)}: {len(batch)} texts, {batch_tokens:,} tokens")
        start_time = time.time()
        
        embeddings = make_api_call_with_retry(voyage_client, batch)
        
        elapsed = time.time() - start_time
        print(f"      ✓ Completed ({elapsed:.1f}s elapsed)")
        
        all_embeddings.extend(embeddings)
        batch_start_indices.append(current_index)
        current_index += len(batch)
    
    print(f"\n   ✅ Generated {len(all_embeddings)} embeddings")
    
    # Verify
    if all_embeddings:
        embedding_dim = len(all_embeddings[0])
        print(f"   ℹ️  Embedding dimension: {embedding_dim}")
    
    # Add to ChromaDB
    print("\n7. Uploading to ChromaDB...")
    collection.add(
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    final_count = collection.count()
    print(f"   ✅ Uploaded {final_count} documents")
    
    # Test search
    print("\n8. Testing search...")
    test_query = "What is a deductible?"
    test_embedding = voyage_client.embed(
        [test_query],
        model="voyage-3",
        input_type="query"
    ).embeddings[0]
    
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=3,
        include=['documents', 'distances']
    )
    
    if results['documents'][0]:
        print(f"   ✅ Search test successful!")
        print(f"   Top result preview: {results['documents'][0][0][:100]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ VECTOR DATABASE CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  - Documents: {final_count}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Batches processed: {len(batches)}")
    print(f"  - Embedding dimension: {embedding_dim if all_embeddings else 'N/A'}")
    print(f"\n✅ Ready to run: python day6_rag.py")

if __name__ == "__main__":
    main()