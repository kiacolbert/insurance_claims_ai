"""
Day 7: FastAPI REST API for Insurance Claims RAG System

Endpoints:
- GET  /health       - Health check
- POST /ask          - Ask a question (RAG)
- GET  /cache/stats  - Cache statistics
- DELETE /cache      - Clear cache
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
from datetime import datetime

# Import your existing RAG components
# You'll need to adapt these imports based on your actual file structure
try:
    from day6_rag import ask_with_rag, retrieve_context
    from src.cache import get_cache
    from src.cost_tracker import get_tracker
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RAG components not found. Using mock mode.")
    RAG_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claims AI API",
    description="RAG-powered Q&A system for insurance policies",
    version="1.0.0"
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
    
    class Config:
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
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available. Check server logs."
        )
    
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
        
        # Build sources list
        sources = [
            Source(
                document=ctx['metadata'].get('source', 'Unknown'),
                chunk_id=ctx['metadata'].get('chunk_id', 'unknown'),
                similarity=round(ctx['similarity'], 4),
                text=ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text']
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
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
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

# ===== STARTUP/SHUTDOWN EVENTS =====

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("\n" + "="*60)
    print("🚀 Insurance Claims AI API Starting...")
    print("="*60)
    print(f"RAG System: {'✅ Available' if RAG_AVAILABLE else '❌ Not Available'}")
    print(f"Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    print("\n👋 Shutting down Insurance Claims AI API\n")

# ===== RUN SERVER =====

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    print("Visit: http://localhost:8000/docs for interactive API docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev only)
    )