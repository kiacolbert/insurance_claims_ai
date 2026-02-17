"""
Embedding generation module using Voyage AI with robust rate limiting.

Handles Voyage AI rate limits:
- 3 RPM (Requests Per Minute)
- 10K TPM (Tokens Per Minute)
"""

import voyageai
import os
import time
import tiktoken
from typing import List
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using Voyage AI with automatic rate limiting and retry."""
    
    # Rate limit constants - VERY conservative to avoid errors
    MAX_TOKENS_PER_REQUEST = 2_000  # Well under 10K TPM limit
    REQUESTS_PER_MINUTE = 2  # Under 3 RPM limit
    SECONDS_BETWEEN_REQUESTS = 30  # 60 seconds / 2 requests = 30s spacing
    
    def __init__(self, model: str = "voyage-2"):
        """
        Initialize embedding generator.
        
        Args:
            model: Voyage AI model to use
        """
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Track usage and timing
        self.total_requests = 0
        self.total_tokens = 0
        self.last_request_time = 0
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _wait_for_rate_limit(self):
        """
        Ensure we respect rate limits by waiting if necessary.
        
        Strategy: Always wait SECONDS_BETWEEN_REQUESTS between calls.
        This is the simplest, most reliable approach.
        """
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            wait_time = self.SECONDS_BETWEEN_REQUESTS - elapsed
            
            if wait_time > 0:
                print(f"      ⏳ Waiting {wait_time:.1f}s for rate limit...")
                time.sleep(wait_time)
    
    def _make_api_call_with_retry(
        self, 
        texts: List[str], 
        input_type: str,
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Make API call with exponential backoff retry.
        
        If we hit rate limit, wait and retry with increasing delays.
        
        Args:
            texts: List of texts to embed
            input_type: "document" or "query"
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of embeddings
            
        Raises:
            Exception if all retries fail
        """
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making request
                self._wait_for_rate_limit()
                
                # Make the API call
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type=input_type
                )
                
                # Success! Track timing and usage
                self.last_request_time = time.time()
                self.total_requests += 1
                tokens_used = sum(self.count_tokens(t) for t in texts)
                self.total_tokens += tokens_used
                
                return result.embeddings
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff: 30s, 60s, 120s
                        wait_time = self.SECONDS_BETWEEN_REQUESTS * (2 ** attempt)
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
    
    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Split texts into batches that respect token limits.
        
        Very conservative batching to avoid rate limits.
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = self.count_tokens(text)
            
            # If single text exceeds limit, truncate it
            if text_tokens > self.MAX_TOKENS_PER_REQUEST:
                print(f"      ⚠️  Text with {text_tokens} tokens truncated to {self.MAX_TOKENS_PER_REQUEST}")
                tokens = self.encoding.encode(text)[:self.MAX_TOKENS_PER_REQUEST]
                text = self.encoding.decode(tokens)
                text_tokens = self.MAX_TOKENS_PER_REQUEST
            
            # If adding this text would exceed limit, start new batch
            if current_tokens + text_tokens > self.MAX_TOKENS_PER_REQUEST and current_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with automatic batching and rate limiting.
        
        This will be SLOW but RELIABLE due to conservative rate limiting.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to print progress updates
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Split into batches
        batches = self._batch_texts(texts)
        
        if show_progress:
            total_tokens = sum(self.count_tokens(t) for t in texts)
            estimated_time = len(batches) * self.SECONDS_BETWEEN_REQUESTS
            print(f"   📦 Processing {len(texts)} texts in {len(batches)} batches")
            print(f"   📊 Total tokens: {total_tokens:,}")
            print(f"   🐌 Estimated time: ~{estimated_time // 60}m {estimated_time % 60}s (conservative rate limiting)")
            print(f"   💡 Tip: This is slow to avoid rate limit errors!\n")
        
        all_embeddings = []
        start_time = time.time()
        
        for i, batch in enumerate(batches, 1):
            if show_progress:
                batch_tokens = sum(self.count_tokens(t) for t in batch)
                print(f"   🔄 Batch {i}/{len(batches)}: {len(batch)} texts, {batch_tokens:,} tokens")
            
            # Make API call with retry logic
            embeddings = self._make_api_call_with_retry(batch, input_type="document")
            
            all_embeddings.extend(embeddings)
            
            if show_progress:
                elapsed = time.time() - start_time
                print(f"      ✓ Completed ({elapsed:.1f}s elapsed)\n")
        
        total_elapsed = time.time() - start_time
        if show_progress:
            print(f"   ✅ All batches complete in {total_elapsed:.1f}s\n")
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = self._make_api_call_with_retry([query], input_type="query")
        return result[0]
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics for this session."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / self.total_requests if self.total_requests > 0 else 0
        }


# Quick test function
def test_embeddings():
    """Test the embedding generator with rate limiting."""
    generator = EmbeddingGenerator()
    
    # Start with a small test
    test_texts = [
        "Python is a programming language.",
        "Machine learning uses algorithms.",
        "Deep learning is powerful."
    ]
    
    print("🧪 Testing Voyage AI Embeddings with Conservative Rate Limiting\n")
    print(f"Using model: {generator.model}")
    print(f"Max tokens per request: {generator.MAX_TOKENS_PER_REQUEST:,}")
    print(f"Seconds between requests: {generator.SECONDS_BETWEEN_REQUESTS}s\n")
    
    # Count total tokens
    total_tokens = sum(generator.count_tokens(t) for t in test_texts)
    print(f"Total tokens to process: {total_tokens:,}\n")
    
    # Generate embeddings
    embeddings = generator.embed_texts(test_texts, show_progress=True)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Each embedding has {len(embeddings[0])} dimensions")
    
    # Show usage stats
    stats = generator.get_usage_stats()
    print(f"\n📊 Usage Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Avg tokens/request: {stats['avg_tokens_per_request']:.0f}")
    
    print("\n✅ Test passed without rate limit errors!")


if __name__ == "__main__":
    test_embeddings()