"""
Experiment with different chunking strategies
"""

from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def load_pdf(pdf_path: str) -> list[dict]:
    """Load PDF with metadata"""
    reader = PdfReader(pdf_path)
    documents = []
    
    for page_num, page in enumerate(reader.pages, 1):
        documents.append({
            'text': page.extract_text(),
            'page': page_num,
            'source': Path(pdf_path).name,
            'total_pages': len(reader.pages)
        })
    
    return documents

def chunk_with_strategy(documents: list[dict], chunk_size: int, overlap: int, strategy_name: str) -> list[dict]:
    """Chunk with specific parameters"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_id = 0
    
    for doc in documents:
        page_chunks = splitter.split_text(doc['text'])
        
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'source': doc['source'],
                'page': doc['page'],
                'total_pages': doc['total_pages'],
                'chunk_on_page': i + 1,
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'strategy': strategy_name
            })
            chunk_id += 1
    
    return chunks

def analyze_chunks(chunks: list[dict]) -> dict:
    """Analyze chunk statistics"""
    char_counts = [c['char_count'] for c in chunks]
    word_counts = [c['word_count'] for c in chunks]
    
    return {
        'total_chunks': len(chunks),
        'avg_chars': sum(char_counts) / len(char_counts),
        'min_chars': min(char_counts),
        'max_chars': max(char_counts),
        'avg_words': sum(word_counts) / len(word_counts),
        'chunks_per_page': len(chunks) / chunks[0]['total_pages'] if chunks else 0
    }

def test_retrieval(chunks: list[dict], test_queries: list[str]) -> dict:
    """Simple keyword retrieval test"""
    results = {}
    
    for query in test_queries:
        query_words = set(query.lower().split())
        
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        top_3 = [c for _, c in scored[:3]]
        
        results[query] = {
            'matches': len(scored),
            'top_chunks': [
                f"Chunk {c['chunk_id']} (page {c['page']}, {c['word_count']} words)"
                for c in top_3
            ]
        }
    
    return results

def main():
    # Load documents
    docs_path = Path("docs")
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  Add PDFs to docs/ first")
        return
    
    documents = []
    for pdf in pdf_files:
        documents.extend(load_pdf(pdf))
    
    print(f"📄 Loaded {len(documents)} pages from {len(pdf_files)} files\n")
    
    # Chunking strategies to test
    strategies = [
        ('Small (500/100)', 500, 100),
        ('Medium (1000/200)', 1000, 200),
        ('Large (1500/200)', 1500, 200),
        ('No Overlap (1000/0)', 1000, 0)
    ]
    
    # Test queries
    test_queries = [
        "collision deductible",
        "file a claim",
        "what is excluded",
        "uninsured motorist"
    ]
    
    print("=" * 70)
    print("CHUNKING EXPERIMENTS")
    print("=" * 70)
    
    all_results = {}
    
    for strategy_name, chunk_size, overlap in strategies:
        print(f"\n📦 {strategy_name}")
        print("-" * 70)
        
        # Chunk
        chunks = chunk_with_strategy(documents, chunk_size, overlap, strategy_name)
        
        # Analyze
        stats = analyze_chunks(chunks)
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Avg size: {stats['avg_chars']:.0f} chars ({stats['avg_words']:.0f} words)")
        print(f"  Range: {stats['min_chars']}-{stats['max_chars']} chars")
        print(f"  Per page: {stats['chunks_per_page']:.1f}")
        
        # Test retrieval
        retrieval = test_retrieval(chunks, test_queries)
        print(f"\n  Retrieval test:")
        for query, result in retrieval.items():
            print(f"    '{query}': {result['matches']} matches")
            if result['top_chunks']:
                print(f"      Top: {result['top_chunks'][0]}")
        
        all_results[strategy_name] = {
            'stats': stats,
            'retrieval': retrieval,
            'sample_chunk': chunks[0]['text'][:200] + "..."
        }
    
    # Save results
    with open('chunk_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 Full results saved to chunk_experiments.json")
    print("=" * 70)
    
    # Recommendation
    print("\n💡 Recommendation:")
    print("  Medium (1000/200) - Best balance for most use cases")
    print("  - Good context per chunk")
    print("  - Overlap preserves sentence boundaries")
    print("  - ~2-3 chunks per page")

if __name__ == "__main__":
    main()