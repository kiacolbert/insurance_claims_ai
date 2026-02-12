# day6_rag.py
"""
Day 6: Complete RAG Q&A System
Combines vector search + Claude for intelligent answers
"""

from anthropic import Anthropic
import chromadb
from voyageai import Client as VoyageClient
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# Initialize clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
voyage_client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("insurance_docs")

print(f"✅ Connected to database ({collection.count()} documents)")

def retrieve_context(question: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant chunks for a question
    
    Returns list of dicts with:
        - content: the chunk text
        - metadata: source info
        - similarity: relevance score
    """
    # Generate embedding for question
    query_embedding = voyage_client.embed(
        [question],
        model="voyage-3",
        input_type="query"
    ).embeddings[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Format results
    contexts = []
    for i in range(len(results['documents'][0])):
        contexts.append({
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'similarity': 1 - results['distances'][0][i]
        })
    
    return contexts

def build_prompt(question: str, contexts: list[dict]) -> str:
    """
    Build the prompt with retrieved context
    """
    # Format context sections
    context_text = "\n\n".join([
        f"--- Context {i+1} ---\n{ctx['content']}"
        for i, ctx in enumerate(contexts)
    ])
    
    prompt = f"""You are an insurance policy assistant. Answer the user's question based ONLY on the provided context from insurance documents.

<context>
{context_text}
</context>

<question>
{question}
</question>

<instructions>
- Answer clearly and concisely
- Use ONLY information from the provided context
- If the context doesn't contain relevant information, say "I don't have that information in the provided documents"
- Cite which context section you used (e.g., "According to Context 1...")
- Be helpful but accurate - don't make assumptions beyond the context
</instructions>

Answer:"""
    
    return prompt

def ask_with_rag(question: str, stream: bool = True) -> str:
    """
    Ask a question using RAG system
    """
    print(f"\n{'='*70}")
    print(f"🔍 Question: {question}")
    print(f"{'='*70}")
    
    # Step 1: Retrieve relevant context
    print("\n📚 Searching knowledge base...")
    contexts = retrieve_context(question, top_k=3)
    
    # Show what we found
    print(f"✅ Found {len(contexts)} relevant documents:\n")
    for i, ctx in enumerate(contexts, 1):
        source = ctx['metadata'].get('source', 'Unknown')
        similarity = ctx['similarity']
        print(f"  {i}. {source}")
        print(f"     Relevance: {similarity:.3f}")
        print(f"     Preview: {ctx['content'][:100]}...\n")
    
    # Step 2: Build prompt with context
    prompt = build_prompt(question, contexts)
    
    # Step 3: Get answer from Claude
    print("🤖 Claude is answering...\n")
    print("-" * 70)
    
    if stream:
        # Streaming response
        full_response = ""
        with anthropic_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print()  # New line after streaming
    else:
        # Non-streaming response
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        full_response = response.content[0].text
        print(full_response)
    
    print("-" * 70)
    return full_response

def main():
    """Interactive RAG Q&A session"""
    print("=" * 70)
    print("INSURANCE POLICY RAG Q&A SYSTEM")
    print("=" * 70)
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'stream on/off' to toggle streaming")
    print("  - 'quit' to exit\n")
    
    stream_mode = True
    
    # Sample questions to try
    print("💡 Try these questions:")
    print("  - What is term life insurance?")
    print("  - Should I buy whole life insurance?")
    print("  - What does long-term care insurance cover?")
    print("  - What is a deductible?\n")
    
    while True:
        try:
            user_input = input("❓ Your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if user_input.lower().startswith('stream'):
                if 'off' in user_input.lower():
                    stream_mode = False
                    print("✅ Streaming disabled\n")
                else:
                    stream_mode = True
                    print("✅ Streaming enabled\n")
                continue
            
            # Process the question
            answer = ask_with_rag(user_input, stream=stream_mode)
            print()  # Extra line for readability
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()