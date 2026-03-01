# src/insurance_claims_ai/cache.py (upgraded)
import hashlib
import json
import os
import redis
from typing import Optional, Dict, Any

class RedisCache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600  # 1 hour
    
    def _make_key(self, question: str) -> str:
        normalized = question.lower().strip()
        return f"claims:qa:{hashlib.md5(normalized.encode()).hexdigest()}"
    
    def get(self, question: str) -> Optional[str]:
        key = self._make_key(question)
        return self.client.get(key)  # returns None if missing or expired
    
    def set(self, question: str, answer: str, ttl: int = None):
        key = self._make_key(question)
        self.client.setex(key, ttl or self.default_ttl, answer)
    
    def stats(self) -> Dict[str, Any]:
        info = self.client.info("stats")
        return {
            "hits": info["keyspace_hits"],
            "misses": info["keyspace_misses"],
            "hit_rate": round(
                info["keyspace_hits"] / max(info["keyspace_hits"] + info["keyspace_misses"], 1) * 100, 2
            ),
            "total_keys": self.client.dbsize()
        }
    
    def clear(self):
        self.client.flushdb()

# Fallback to in-memory if Redis is unavailable
def get_cache():
    try:
        cache = RedisCache()
        cache.client.ping()  # test connection
        return cache
    except Exception:
        print("⚠️  Redis unavailable, falling back to in-memory cache")
        from .simple_cache import SimpleCache 
        return SimpleCache()