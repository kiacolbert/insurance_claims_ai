"""
Simple caching layer for LLM responses
Prevents duplicate API calls for similar questions
"""

import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class SimpleCache:
    """In-memory cache (loses data on restart)"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, question: str) -> str:
        """Generate cache key from question"""
        # Normalize: lowercase, strip whitespace
        normalized = question.lower().strip()
        # Hash for consistent key length
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, question: str) -> Optional[str]:
        """Get cached answer if exists"""
        key = self._generate_key(question)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired (5 min TTL)
            if datetime.now() < entry['expires_at']:
                self.hits += 1
                return entry['answer']
            else:
                # Expired, remove it
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, question: str, answer: str, ttl_seconds: int = 300):
        """Cache an answer"""
        key = self._generate_key(question)
        
        self.cache[key] = {
            'question': question,
            'answer': answer,
            'cached_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl_seconds)
        }
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_items': len(self.cache)
        }
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

# Global cache instance
_cache = SimpleCache()

def get_cache() -> SimpleCache:
    """Get the global cache instance"""
    return _cache