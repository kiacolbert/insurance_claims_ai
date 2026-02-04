"""
Day 2: Q&A with caching
Same as Day 1 but now with cache layer
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import time
from src.cache import get_cache

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cache = get_cache()

# Same policy document as Day 1
POLICY_DOCUMENT = """
AUTO INSURANCE POLICY - Summary
[... same content as Day 1 ...]
"""

def ask_question(question: str) -> tuple[str, bool, float]:
    """
    Ask a question about the insurance policy
    
    Returns:
        tuple: (answer, was_cached, response_time_ms)
    """
    start_time = time.time()
    
    # Check cache first
    cached_answer = cache.get(question)
    if cached_answer:
        response_time = (time.time() - start_time) * 1000
        return cached_answer, True, response_time
    
    # Not in cache, call LLM
    prompt = f"""You are an insurance policy assistant helping customers understand their coverage.

Policy Document:
{POLICY_DOCUMENT}

Customer Question: {question}

Instructions:
- Answer based ONLY on the policy document provided
- Be clear and concise
- If the information isn't in the policy, say "This information is not covered in your policy document"
- Cite specific sections when possible

Answer:"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.content[0].text
    
    # Cache the answer
    cache.set(question, answer, ttl_seconds=300)  # 5 min TTL
    
    response_time = (time.time() - start_time) * 1000
    return answer, False, response_time

def main():
    """Interactive Q&A session with caching stats"""
    print("=" * 60)
    print("INSURANCE POLICY Q&A ASSISTANT (with caching)")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type question to ask")
    print("  - 'stats' to see cache statistics")
    print("  - 'clear' to clear cache")
    print("  - 'quit' to exit\n")
    
    while True:
        user_input = input("❓ Question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n📊 Final Statistics:")
            stats = cache.stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'stats':
            print("\n📊 Cache Statistics:")
            stats = cache.stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            continue
        
        if user_input.lower() == 'clear':
            cache.clear()
            print("✅ Cache cleared!\n")
            continue
        
        if not user_input:
            continue
        
        print("\n🤔 Thinking...\n")
        answer, was_cached, response_time = ask_question(user_input)
        
        cache_status = "💾 CACHED" if was_cached else "🌐 API CALL"
        print(f"💡 Answer:\n{answer}\n")
        print(f"{cache_status} | ⏱️  {response_time:.0f}ms")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()