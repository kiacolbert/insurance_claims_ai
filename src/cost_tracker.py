"""
Cost Tracker for Insurance Claims AI

Tracks API usage, token consumption, and costs with cache savings analysis.
Works with Anthropic's Claude API.
"""

from typing import Dict, Any
from datetime import datetime

class CostTracker:
    """
    Track costs and token usage for LLM API calls
    
    Anthropic Claude Pricing (as of Feb 2026):
    - Claude Haiku: $0.25 per million input tokens, $1.25 per million output tokens
    - Claude Sonnet: $3 per million input tokens, $15 per million output tokens
    """
    
    def __init__(self, model: str = "claude-sonnet-4"):
        self.model = model
        
        # Pricing per million tokens
        self.pricing = {
            "claude-haiku-4": {
                "input": 0.25,
                "output": 1.25
            },
            "claude-sonnet-4": {
                "input": 3.00,
                "output": 15.00
            },
            "claude-opus-4": {
                "input": 15.00,
                "output": 75.00
            }
        }
        
        # Counters
        self.total_requests = 0
        self.cached_requests = 0
        self.api_calls = 0
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        self.total_cost_usd = 0.0
        self.cost_without_cache_usd = 0.0
    
    def track_request(
        self, 
        input_tokens: int = 0,
        output_tokens: int = 0, 
        was_cached: bool = False
    ):
        """
        Track a single request
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            was_cached: Whether this was a cache hit (no API call)
        """
        self.total_requests += 1
        
        if was_cached:
            self.cached_requests += 1
            # Cache hit - no cost incurred
            # But track what it WOULD have cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            self.cost_without_cache_usd += cost
        else:
            self.api_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # Calculate and add cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            self.total_cost_usd += cost
            self.cost_without_cache_usd += cost
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts"""
        model_pricing = self.pricing.get(self.model, self.pricing["claude-sonnet-4"])
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics
        
        Returns:
            Dictionary with all metrics
        """
        savings = self.cost_without_cache_usd - self.total_cost_usd
        savings_percent = (
            (savings / self.cost_without_cache_usd * 100)
            if self.cost_without_cache_usd > 0 
            else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "cached_requests": self.cached_requests,
            "api_calls": self.api_calls,
            "cache_hit_rate_percent": round(
                (self.cached_requests / self.total_requests * 100)
                if self.total_requests > 0 else 0,
                2
            ),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "cost_without_cache_usd": round(self.cost_without_cache_usd, 4),
            "savings_usd": round(savings, 4),
            "savings_percent": round(savings_percent, 2),
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_stats(self):
        """Print formatted statistics to console"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("💰 COST ANALYSIS")
        print("="*60)
        
        print(f"\n📊 Usage:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Cached: {stats['cached_requests']} ({stats['cache_hit_rate_percent']}%)")
        print(f"  API Calls: {stats['api_calls']}")
        
        print(f"\n🎯 Tokens:")
        print(f"  Input: {stats['total_input_tokens']:,}")
        print(f"  Output: {stats['total_output_tokens']:,}")
        print(f"  Total: {stats['total_tokens']:,}")
        
        print(f"\n💵 Costs:")
        print(f"  Actual Cost: ${stats['total_cost_usd']:.4f}")
        print(f"  Without Cache: ${stats['cost_without_cache_usd']:.4f}")
        print(f"  Savings: ${stats['savings_usd']:.4f} ({stats['savings_percent']}%)")
        
        print(f"\n🤖 Model: {stats['model']}")
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all counters"""
        self.total_requests = 0
        self.cached_requests = 0
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.cost_without_cache_usd = 0.0


# Global tracker instance
_tracker = CostTracker()

def get_tracker() -> CostTracker:
    """Get the global cost tracker instance"""
    return _tracker


# Example usage
if __name__ == "__main__":
    print("Cost Tracker Example\n")
    
    tracker = CostTracker(model="claude-sonnet-4")
    
    # Simulate some requests
    print("Simulating requests...")
    
    # Request 1: Uncached (API call)
    tracker.track_request(input_tokens=300, output_tokens=50, was_cached=False)
    
    # Request 2: Cached (no API call)
    tracker.track_request(input_tokens=300, output_tokens=50, was_cached=True)
    
    # Request 3: Uncached
    tracker.track_request(input_tokens=400, output_tokens=75, was_cached=False)
    
    # Request 4: Cached
    tracker.track_request(input_tokens=400, output_tokens=75, was_cached=True)
    
    # Request 5: Uncached
    tracker.track_request(input_tokens=350, output_tokens=60, was_cached=False)
    
    # Print stats
    tracker.print_stats()
    
    # Get stats as dictionary
    stats = tracker.get_stats()
    print("Stats as dict:")
    import json
    print(json.dumps(stats, indent=2))