"""
Day 4: Chunk documents for better retrieval
Split large PDFs into smaller, manageable pieces
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.cache import get_cache

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cache = get_cache()

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from PDF with page metadata
    
    Returns:
        List of dicts with 'text', 'page', 'source'
    """
    reader = PdfReader(pdf_path)
    documents = []
    
    for page_num, page in enumerate(reader.pages, 1):
        documents.append({
            'text': page.extract_text(),
            'page': page_num,
            'source': Path(pdf_path).name
        })
    
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into smaller chunks
    
    Args:
        documents: List of page dicts from load_pdf()
        
    Returns:
        List of chunk dicts with text, metadata, chunk_id
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Characters per chunk
        chunk_overlap=100,     # Overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_id = 0
    
    for doc in documents:
        # Split the page text
        page_chunks = splitter.split_text(doc['text'])
        
        # Add metadata to each chunk
        for chunk_text in page_chunks:
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'source': doc['source'],
                'page': doc['page'],
                'char_count': len(chunk_text)
            })
            chunk_id += 1
    
    return chunks

def load_and_chunk_all(docs_dir: str = "docs") -> list[dict]:
    """Load all PDFs and chunk them"""
    docs_path = Path(docs_dir)
    
    if not docs_path.exists() or not list(docs_path.glob("*.pdf")):
        print(f"⚠️  No PDFs in {docs_dir}/")
        return []
    
    all_chunks = []
    pdf_files = list(docs_path.glob("*.pdf"))
    
    print(f"📄 Processing {len(pdf_files)} PDF files...\n")
    
    for pdf_file in pdf_files:
        print(f"  {pdf_file.name}...", end=" ")
        
        # Load pages
        documents = load_pdf(pdf_file)
        
        # Chunk pages
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)
        
        print(f"✓ ({len(documents)} pages → {len(chunks)} chunks)")
    
    print(f"\n📦 Total: {len(all_chunks)} chunks\n")
    return all_chunks

def find_relevant_chunks(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Simple keyword-based retrieval (will be replaced with embeddings in Day 5)
    
    Args:
        question: User's question
        chunks: All document chunks
        top_k: Number of chunks to return
        
    Returns:
        Top K most relevant chunks
    """
    # Score chunks by keyword overlap
    question_words = set(question.lower().split())
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk['text'].lower().split())
        overlap = len(question_words & chunk_words)
        
        if overlap > 0:
            scored_chunks.append((overlap, chunk))
    
    # Sort by score, return top K
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k]]

def ask_question(question: str, chunks: list[dict]) -> tuple[str, bool, float, int, list[dict]]:
    """
    Ask question using chunked retrieval
    
    Returns:
        (answer, was_cached, response_time_ms, tokens_used, relevant_chunks)
    """
    start_time = time.time()
    
    # Check cache
    cached_answer = cache.get(question)
    if cached_answer:
        response_time = (time.time() - start_time) * 1000
        return cached_answer, True, response_time, 0, []
    
    # Retrieve relevant chunks
    relevant_chunks = find_relevant_chunks(question, chunks, top_k=3)
    
    # Build context from chunks
    context = "\n\n".join([
        f"[{chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    # Ask LLM
    prompt = f"""You are an insurance policy assistant.

Relevant Policy Sections:
{context}

Customer Question: {question}

Instructions:
- Answer based ONLY on the sections provided above
- Cite the source and page number
- If information isn't in the sections, say so

Answer:"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    
    cache.set(question, answer, ttl_seconds=300)
    
    response_time = (time.time() - start_time) * 1000
    return answer, False, response_time, tokens_used, relevant_chunks

def main():
    """Interactive Q&A with chunked retrieval"""
    
    # Load and chunk all documents
    chunks = load_and_chunk_all("docs")
    
    if not chunks:
        print("❌ No documents loaded. Add PDFs to docs/ directory.")
        return
    
    print("=" * 60)
    print("INSURANCE Q&A - CHUNKED RETRIEVAL")
    print("=" * 60)
    print("\nCommands: question | 'stats' | 'clear' | 'quit'\n")
    
    total_cost = 0.0
    
    while True:
        user_input = input("❓ Question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n💰 Total cost: ${total_cost:.4f}")
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'stats':
            print(f"\n📊 Stats:")
            print(f"  Chunks loaded: {len(chunks)}")
            print(f"  Session cost: ${total_cost:.4f}")
            for key, value in cache.stats().items():
                print(f"  {key}: {value}")
            print()
            continue
        
        if user_input.lower() == 'clear':
            cache.clear()
            print("✅ Cache cleared!\n")
            continue
        
        if not user_input:
            continue
        
        print("\n🔍 Searching...\n")
        answer, was_cached, response_time, tokens, relevant = ask_question(
            user_input, chunks
        )
        
        cost = (tokens * 0.5) / 1_000_000
        total_cost += cost
        
        print(f"💡 Answer:\n{answer}\n")
        
        if not was_cached and relevant:
            print("📚 Sources:")
            for chunk in relevant:
                print(f"  • {chunk['source']}, Page {chunk['page']} (chunk {chunk['chunk_id']})")
        
        cache_status = "💾 CACHED" if was_cached else "🌐 API"
        print(f"\n{cache_status} | ⏱️  {response_time:.0f}ms", end="")
        if not was_cached:
            print(f" | 🔢 {tokens} tokens | 💰 ${cost:.4f}")
        else:
            print(" | FREE")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()