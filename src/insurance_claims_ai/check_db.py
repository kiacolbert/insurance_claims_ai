# check_db.py
import chromadb

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

# List all collections
collections = chroma_client.list_collections()

print(f"Found {len(collections)} collections:")
for col in collections:
    print(f"  - {col.name} ({col.count()} items)")