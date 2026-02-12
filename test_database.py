# test_database.py
"""
Quick test to verify your vector database is ready
"""

import chromadb
from voyageai import Client as VoyageClient
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("insurance_docs")

print(f"✅ Database found!")
print(f"📊 Documents in database: {collection.count()}")

# Test a search
voyage_client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))

test_query = "What is a deductible?"
print(f"\n🔍 Testing search for: '{test_query}'")

query_embedding = voyage_client.embed(
    [test_query],
    model="voyage-3",
    input_type="query"
).embeddings[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=['documents', 'distances', 'metadatas']
)

print(f"\n📚 Found {len(results['documents'][0])} results:")
for i, (doc, dist, meta) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
), 1):
    similarity = 1 - dist
    print(f"\n{i}. Similarity: {similarity:.3f}")
    print(f"   Source: {meta.get('source', 'Unknown')}")
    print(f"   Preview: {doc[:150]}...")

print("\n✅ Database is working perfectly!")
print("🚀 Ready for Day 6!")