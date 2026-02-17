"""
Day 7: FastAPI REST API for Insurance Claims RAG System

Endpoints:
- GET  /health        - Health check
- POST /ask          - Ask a question (RAG)
- GET  /cache/stats  - Cache statistics
- DELETE /cache      - Clear cache
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import time
from datetime import datetime

# Import cache and cost tracking (standalone modules)
from insurance_claims_ai.cache import get_cache
from insurance_claims_ai.cost_tracker import get_tracker

# Try to import RAG components (optional - API works without them in demo mode)
try:
    from insurance_claims_ai.day6_rag import ask_with_rag, retrieve_context
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RAG components not found. API running in DEMO mode.")
    print("    The API will work, but /ask endpoint will return mock responses.")
    print("    To enable full RAG, make sure you have day6_rag.py")
    RAG_AVAILABLE = False
    
    # Mock functions for demo mode
    def retrieve_context(question: str, top_k: int = 3):
        """Mock context retrieval"""
        return [
            {
                'context': 'Your collision deductible is $500 per accident.',
                'metadata': {'source': 'auto_policy_2024.pdf', 'chunk_id': 'chunk_12'},
                'similarity': 0.89
            },
            {
                'context': 'Comprehensive coverage has a $250 deductible.',
                'metadata': {'source': 'auto_policy_2024.pdf', 'chunk_id': 'chunk_15'},
                'similarity': 0.75
            },
            {
                'context': 'You can file claims by calling 1-800-CLAIMS or online.',
                'metadata': {'source': 'claims_guide.pdf', 'chunk_id': 'chunk_3'},
                'similarity': 0.68
            }
        ]
    
    def ask_with_rag(question: str, stream: bool = False):
        """Mock RAG response"""
        import random
        responses = [
            "Based on your policy, your collision deductible is $500 per accident.",
            "To file a claim, you can call 1-800-CLAIMS or visit our website within 24 hours of the incident.",
            "Yes, your policy includes uninsured motorist coverage at no additional cost.",
            "Your comprehensive coverage includes a $250 deductible for incidents like theft, vandalism, or weather damage."
        ]
        return random.choice(responses)

# ===== LIFESPAN EVENT HANDLER =====

# Check if we're in test mode
IS_TEST = os.getenv("TESTING", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    print("🚀 Starting up...")
    # Initialize your RAG system, load models, etc.
    yield
    # Shutdown
    print("🛑 Shutting down...")
    # Cleanup resources

# Only use lifespan in production
if IS_TEST:
    app = FastAPI(
        title="Insurance Claims AI API",
        description="RAG-powered Q&A system for insurance policies",
        version="1.0.0"
    )
else:
    app = FastAPI(
        title="Insurance Claims AI API",
        description="RAG-powered Q&A system for insurance policies",
        version="1.0.0",
        lifespan=lifespan
    )

# Add CORS middleware (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== REQUEST/RESPONSE MODELS =====

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=3, max_length=500, 
                         description="The question to ask")
    use_cache: bool = Field(True, description="Whether to use cached responses")
    top_k: int = Field(3, ge=1, le=10, 
                      description="Number of context chunks to retrieve")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "question": "What is my collision deductible?",
                "use_cache": True,
                "top_k": 3
            }
        }

class Source(BaseModel):
    """Source document information"""
    document: str
    chunk_id: str
    similarity: float
    text: str

class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    sources: List[Source]
    cached: bool
    response_time_ms: int
    timestamp: str
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    rag_available: bool
    cache_size: int
    
class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate_percent: float
    cached_items: int

# ===== API ENDPOINTS =====

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Insurance Claims AI API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns system status and basic metrics
    """
    cache = get_cache() if RAG_AVAILABLE else None
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        rag_available=RAG_AVAILABLE,
        cache_size=len(cache.cache) if cache else 0
    )

@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question about insurance policies using RAG
    
    - **question**: Your question about the insurance policy
    - **use_cache**: Whether to use cached responses (default: true)
    - **top_k**: Number of relevant document chunks to retrieve (1-10)
    
    Returns answer with source citations and metadata.
    """
    start_time = time.time()
    
    try:
        # Check cache first if enabled
        cache = get_cache()
        cached_answer = None
        
        if request.use_cache:
            cached_answer = cache.get(request.question)
        
        if cached_answer:
            # Return cached response
            response_time = int((time.time() - start_time) * 1000)
            
            return AnswerResponse(
                answer=cached_answer,
                sources=[],  # Cached responses don't include sources
                cached=True,
                response_time_ms=response_time,
                timestamp=datetime.now().isoformat()
            )
        
        # Not cached - perform RAG
        # First, retrieve context
        contexts = retrieve_context(request.question, top_k=request.top_k)
        
        # Generate answer with RAG
        answer = ask_with_rag(request.question, stream=False)
        
        # Cache the answer
        cache.set(request.question, answer)
        print(f"context: {contexts[0]}")
        
        # Build sources list
        sources = [
            Source(
                document=ctx['metadata'].get('source', 'Unknown'),
                chunk_id=ctx['metadata'].get('chunk_id', 'unknown'),
                similarity=round(ctx['similarity'], 4),
                text=ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['text']
            )
            for ctx in contexts
        ]
        
        response_time = int((time.time() - start_time) * 1000)
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            cached=False,
            response_time_ms=response_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"\n❌ ERROR in /ask endpoint:")
        print(f"   {error_details}\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {type(e).__name__}: {str(e)}"
        )

@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
async def get_cache_stats():
    """
    Get cache statistics
    
    Returns metrics about cache performance including hit rate.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    cache = get_cache()
    stats = cache.stats()
    
    return CacheStatsResponse(**stats)

@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """
    Clear the entire cache
    
    Use with caution - this will remove all cached responses.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    cache = get_cache()
    items_cleared = len(cache.cache)
    cache.clear()
    
    return {
        "status": "success",
        "message": f"Cleared {items_cleared} cached items",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cost/stats", tags=["Monitoring"])
async def get_cost_stats():
    """
    Get cost and token usage statistics
    
    Returns detailed metrics about API costs and savings from caching.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracker not available")
    
    tracker = get_tracker()
    stats = tracker.get_stats()
    
    return {
        "total_requests": stats['total_requests'],
        "cached_requests": stats['cached_requests'],
        "api_calls": stats['api_calls'],
        "total_cost_usd": stats['total_cost_usd'],
        "cost_without_cache_usd": stats['cost_without_cache_usd'],
        "savings_usd": stats['savings_usd'],
        "savings_percent": stats['savings_percent'],
        "total_input_tokens": stats['total_input_tokens'],
        "total_output_tokens": stats['total_output_tokens'],
        "timestamp": datetime.now().isoformat()
    }

# ===== RUN SERVER =====

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    print("Visit: http://localhost:8000/docs for interactive API docs")
    
    uvicorn.run(
        "api:app",  # Import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True  # Now reload works properly
    )