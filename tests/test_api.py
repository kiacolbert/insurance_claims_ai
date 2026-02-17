"""
Pytest test suite for Insurance Claims AI API

Run with: pytest test_api.py -v
"""

import os
os.environ["TESTING"] = "true" # Set testing environment variable

os.environ["TESTING"] = "true"
os.environ["CHROMA_DB_PATH"] = str(__file__.rsplit('/', 1)[0] + "/../chroma_db")

import pytest

# Debug imports
print("=" * 60)
print("DEBUGGING IMPORTS")
print("=" * 60)

try:
    from insurance_claims_ai.api import app
    print(f"✅ Import successful")
    print(f"   Type: {type(app)}")
    print(f"   Module: {type(app).__module__}")
    print(f"   Class: {type(app).__name__}")
    
    # Check if it's actually FastAPI
    from fastapi import FastAPI
    print(f"   Is FastAPI instance? {isinstance(app, FastAPI)}")
    
    # Check for lifespan
    if hasattr(app, 'router'):
        print(f"   Has router: ✅")
    if hasattr(app, 'lifespan_context'):
        print(f"   Has lifespan_context: ✅")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

from fastapi.testclient import TestClient

# Now try creating TestClient
print("\nTrying to create TestClient...")
try:
    client = TestClient(app)
    print("✅ TestClient created successfully!")
except Exception as e:
    print(f"❌ TestClient creation failed: {e}")
    import traceback
    traceback.print_exc()
    
# import pytest
from fastapi.testclient import TestClient
from insurance_claims_ai.api import app

# Create test client
client = TestClient(app)

# ===== FIXTURES =====

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test"""
    client.delete("/cache")
    yield

# ===== HEALTH & ROOT TESTS =====

def test_root_endpoint():
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "operational"

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "rag_available" in data
    assert "cache_size" in data

# ===== ASK ENDPOINT TESTS =====

def test_ask_valid_question():
    """Test asking a valid question"""
    payload = {
        "question": "What is my collision deductible?",
        "use_cache": True,
        "top_k": 3
    }
    
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "cached" in data
    assert "response_time_ms" in data
    assert "timestamp" in data
    assert data["cached"] == False  # First call shouldn't be cached

def test_ask_cached_question():
    """Test that repeated questions are cached"""
    question = "How do I file a claim?"
    payload = {
        "question": question,
        "use_cache": True,
        "top_k": 3
    }
    
    # First call
    response1 = client.post("/ask", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cached"] == False
    
    # Second call (should be cached)
    response2 = client.post("/ask", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cached"] == True
    
    # Cached response should be faster
    assert data2["response_time_ms"] < data1["response_time_ms"]

def test_ask_without_cache():
    """Test asking question with caching disabled"""
    question = "What is covered under liability?"
    payload = {
        "question": question,
        "use_cache": False,
        "top_k": 3
    }
    
    # First call
    response1 = client.post("/ask", json=payload)
    data1 = response1.json()
    
    # Second call should also not be cached
    response2 = client.post("/ask", json=payload)
    data2 = response2.json()
    
    assert data1["cached"] == False
    assert data2["cached"] == False

def test_ask_invalid_question_too_short():
    """Test that questions must be at least 3 characters"""
    payload = {
        "question": "Hi",  # Too short
        "use_cache": True,
        "top_k": 3
    }
    
    response = client.post("/ask", json=payload)
    assert response.status_code == 422  # Validation error

def test_ask_invalid_question_too_long():
    """Test that questions can't be too long"""
    payload = {
        "question": "x" * 501,  # Too long (max 500)
        "use_cache": True,
        "top_k": 3
    }
    
    response = client.post("/ask", json=payload)
    assert response.status_code == 422

def test_ask_invalid_top_k():
    """Test that top_k must be between 1 and 10"""
    # Test top_k = 0
    response1 = client.post("/ask", json={
        "question": "What is my deductible?",
        "top_k": 0
    })
    assert response1.status_code == 422
    
    # Test top_k = 11
    response2 = client.post("/ask", json={
        "question": "What is my deductible?",
        "top_k": 11
    })
    assert response2.status_code == 422

# ===== CACHE TESTS =====

def test_cache_stats():
    """Test cache statistics endpoint"""
    # Make some requests first
    questions = [
        "What is my deductible?",
        "How do I file a claim?",
        "What is my deductible?",  # Repeat
    ]
    
    for q in questions:
        client.post("/ask", json={"question": q})
    
    # Get cache stats
    response = client.get("/cache/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_requests" in data
    assert "cache_hits" in data
    assert "cache_misses" in data
    assert "hit_rate_percent" in data
    assert "cached_items" in data
    
    # We should have 3 total requests, 1 hit, 2 misses
    assert data["total_requests"] == 3
    assert data["cache_hits"] == 1
    assert data["cache_misses"] == 2

def test_clear_cache():
    """Test clearing the cache"""
    # Add some items to cache
    client.post("/ask", json={"question": "Test question 1"})
    client.post("/ask", json={"question": "Test question 2"})
    
    # Verify cache has items
    stats_before = client.get("/cache/stats").json()
    assert stats_before["cached_items"] > 0
    
    # Clear cache
    response = client.delete("/cache")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Verify cache is empty
    stats_after = client.get("/cache/stats").json()
    assert stats_after["cached_items"] == 0

# ===== COST STATS TESTS =====

def test_cost_stats():
    """Test cost statistics endpoint"""
    # Make some requests
    client.post("/ask", json={"question": "What is my deductible?"})
    client.post("/ask", json={"question": "What is my deductible?"})  # Cached
    
    response = client.get("/cost/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_requests" in data
    assert "cached_requests" in data
    assert "api_calls" in data
    assert "total_cost_usd" in data
    assert "savings_usd" in data
    assert "savings_percent" in data
    
    # Should show some savings from caching
    assert data["cached_requests"] > 0
    assert data["savings_percent"] > 0

# ===== EDGE CASES =====

def test_multiple_concurrent_same_question():
    """Test asking same question multiple times"""
    question = "Is racing covered?"
    payload = {"question": question}
    
    # Ask 5 times
    responses = [client.post("/ask", json=payload) for _ in range(5)]
    
    # All should succeed
    for r in responses:
        assert r.status_code == 200
    
    # Only first should be uncached
    assert responses[0].json()["cached"] == False
    for r in responses[1:]:
        assert r.json()["cached"] == True

def test_special_characters_in_question():
    """Test questions with special characters"""
    questions = [
        "What's my deductible?",  # Apostrophe
        "Coverage: $100,000?",     # Dollar sign
        "Email: claims@insure.com?",  # At sign
    ]
    
    for q in questions:
        response = client.post("/ask", json={"question": q})
        assert response.status_code == 200

# ===== PERFORMANCE TESTS =====

def test_response_time():
    """Test that uncached responses are reasonably fast"""
    import time
    
    start = time.time()
    response = client.post("/ask", json={
        "question": "What is my policy number?"
    })
    duration = time.time() - start
    
    assert response.status_code == 200
    # Should respond within 5 seconds (adjust based on your system)
    assert duration < 5.0

def test_cached_response_time():
    """Test that cached responses are very fast"""
    import time
    
    question = "Quick cache test question?"
    
    # First call (uncached)
    client.post("/ask", json={"question": question})
    
    # Second call (cached)
    start = time.time()
    response = client.post("/ask", json={"question": question})
    duration = time.time() - start
    
    assert response.status_code == 200
    assert response.json()["cached"] == True
    # Cached should be very fast (< 100ms)
    assert duration < 0.1

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])