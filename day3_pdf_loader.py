"""
Day 3: Load and query PDF documents
Now loading from actual PDFs instead of hardcoded JSON
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from pypdf import PdfReader
from src.cache import get_cache

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cache = get_cache()

def load_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    reader = PdfReader(pdf_path)
    text = ""
    
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num} ---\n{page_text}\n"
    
    return text

def load_all_documents(docs_dir: str = "docs") -> str:
    """
    Load all PDFs from directory
    
    Args:
        docs_dir: Directory containing PDF files
        
    Returns:
        Combined text from all PDFs
    """
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        docs_path.mkdir()
        print(f"📁 Created {docs_dir}/ directory")
        print(f"⚠️  Add your PDF files to {docs_dir}/ and run again")
        return ""
    
    all_text = []
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {docs_dir}/")
        return ""
    
    print(f"📄 Loading {len(pdf_files)} PDF files...\n")
    
    for pdf_file in pdf_files:
        print(f"  Loading {pdf_file.name}...", end=" ")
        text = load_pdf(pdf_file)
        all_text.append(f"=== {pdf_file.name} ===\n{text}")
        
        # Count pages
        reader = PdfReader(pdf_file)
        print(f"✓ ({len(reader.pages)} pages)")
    
    print()
    return "\n\n".join(all_text)

def ask_question(question: str, knowledge_base: str) -> tuple[str, bool, float, int]:
    """
    Ask a question using PDF knowledge base
    
    Returns:
        tuple: (answer, was_cached, response_time_ms, tokens_used)
    """
    start_time = time.time()
    
    # Check cache
    cached_answer = cache.get(question)
    if cached_answer:
        response_time = (time.time() - start_time) * 1000
        return cached_answer, True, response_time, 0
    
    # Call LLM
    prompt = f"""You are an insurance policy assistant helping customers understand their coverage.

Policy Documents:
{knowledge_base}

Customer Question: {question}

Instructions:
- Answer based ONLY on the policy documents provided
- Be clear and concise
- Cite specific sections or page numbers when possible
- If the information isn't in the documents, say so clearly

Answer:"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    
    # Cache the answer
    cache.set(question, answer, ttl_seconds=300)
    
    response_time = (time.time() - start_time) * 1000
    return answer, False, response_time, tokens_used

def main():
    """Interactive Q&A with PDF documents"""
    
    # Load documents
    knowledge_base = load_all_documents("docs")
    
    if not knowledge_base:
        print("❌ No documents loaded. Exiting.")
        return
    
    print("=" * 60)
    print("INSURANCE POLICY Q&A - PDF EDITION")
    print("=" * 60)
    print("\nCommands: question | 'stats' | 'clear' | 'quit'\n")
    
    total_cost = 0.0
    
    while True:
        user_input = input("❓ Question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n💰 Total session cost: ${total_cost:.4f}")
            print("📊 Cache stats:")
            for key, value in cache.stats().items():
                print(f"  {key}: {value}")
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'stats':
            print("\n📊 Cache Statistics:")
            for key, value in cache.stats().items():
                print(f"  {key}: {value}")
            print(f"💰 Session cost: ${total_cost:.4f}\n")
            continue
        
        if user_input.lower() == 'clear':
            cache.clear()
            print("✅ Cache cleared!\n")
            continue
        
        if not user_input:
            continue
        
        print("\n🤔 Thinking...\n")
        answer, was_cached, response_time, tokens = ask_question(
            user_input, knowledge_base
        )
        
        # Calculate cost (Haiku: $0.25/1M input, $1.25/1M output tokens)
        # Simplified: assume 80/20 input/output split
        cost = (tokens * 0.5) / 1_000_000  # Average cost
        total_cost += cost
        
        cache_status = "💾 CACHED" if was_cached else "🌐 API CALL"
        
        print(f"💡 Answer:\n{answer}\n")
        print(f"{cache_status} | ⏱️  {response_time:.0f}ms", end="")
        if not was_cached:
            print(f" | 🔢 {tokens} tokens | 💰 ${cost:.4f}")
        else:
            print(" | 💰 $0.0000 (FREE!)")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
