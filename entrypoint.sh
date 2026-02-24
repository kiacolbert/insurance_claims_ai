#!/bin/bash
# 1. Check if collection already exists in ChromaDB
# 2. If not: run load_data_once.py → run ingestion.py
# 3. Start the API
set -e
if python -c "from chromadb import PersistentClient; client = PersistentClient(path='./data/chroma_db'); print(client.list_collections())" | grep -q "insurance_docs"; then
    echo "Collection already exists in ChromaDB. Skipping data loading and ingestion."
else
    echo "Collection not found in ChromaDB. Running data loading and ingestion."
    python src/insurance_claims_ai/load_data_once.py
    python src/insurance_claims_ai/ingestion.py
fi

echo "🚀 Starting API..."
exec uvicorn insurance_claims_ai.api:app --host 0.0.0.0 --port 8000 --workers 2