import pytest
import os
from pathlib import Path
from unittest.mock import patch

# Set test environment BEFORE any imports
os.environ["TESTING"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["CHROMA_DB_PATH"] = str(Path(__file__).parent.parent / "test_chroma_db")

MOCK_CONTEXTS = [
    {
        "content": "Your collision deductible is $500 per accident.",
        "metadata": {"source": "policy.pdf", "chunk_id": "chunk_1"},
        "similarity": 0.95
    },
    {
        "content": "To file a claim, call 1-800-CLAIMS within 24 hours.",
        "metadata": {"source": "claims_guide.pdf", "chunk_id": "chunk_2"},
        "similarity": 0.87
    },
    {
        "content": "Comprehensive coverage has a $250 deductible.",
        "metadata": {"source": "policy.pdf", "chunk_id": "chunk_3"},
        "similarity": 0.82
    }
]


def mock_retrieve_context(question: str, top_k: int = 3):
    """Mock retrieve_context to avoid hitting real ChromaDB"""
    return MOCK_CONTEXTS[:top_k]


@pytest.fixture(scope="session")
def client():
    """Create test client with mocked dependencies"""
    from fastapi.testclient import TestClient
    import insurance_claims_ai.api as api_module

    # Mock retrieve_context - prevents hitting real vector DB
    with patch.object(api_module, "retrieve_context", mock_retrieve_context):
        with TestClient(api_module.app) as c:
            yield c

    api_module.app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_cache(client):
    """Clear cache before each test"""
    try:
        client.delete("/cache")
    except:
        pass
    yield


# RAG System fixtures (if you have rag_system.py)
try:
    import shutil
    import chromadb
    from insurance_claims_ai.rag_system import RAGSystem, MockLLMClient, reset_rag_system

    @pytest.fixture(scope="session")
    def test_chroma_path(tmp_path_factory):
        """Create temporary ChromaDB for tests"""
        test_db = tmp_path_factory.mktemp("test_chroma_db")
        os.environ["CHROMA_DB_PATH"] = str(test_db)
        yield test_db
        shutil.rmtree(test_db, ignore_errors=True)

    @pytest.fixture(scope="session")
    def populate_test_db(test_chroma_path):
        """Populate test database with sample documents"""
        client = chromadb.PersistentClient(path=str(test_chroma_path))
        
        try:
            collection = client.get_collection("insurance_docs")
        except:
            collection = client.create_collection("insurance_docs")
        
        collection.add(
            documents=[
                "Your collision deductible is $500 per incident.",
                "To file a claim, call 1-800-CLAIMS within 24 hours.",
                "Liability coverage limit is $100,000 per person.",
                "Comprehensive coverage includes theft and vandalism."
            ],
            ids=["doc1", "doc2", "doc3", "doc4"],
            metadatas=[
                {"source": "policy.pdf", "page": 1},
                {"source": "claims.pdf", "page": 1},
                {"source": "policy.pdf", "page": 2},
                {"source": "policy.pdf", "page": 3}
            ]
        )
        
        yield collection

    @pytest.fixture(scope="function")
    def mock_rag_system(test_chroma_path, populate_test_db):
        """Create RAG system with mock LLM"""
        reset_rag_system()
        
        rag = RAGSystem(
            chroma_path=test_chroma_path,
            llm_client=MockLLMClient()
        )
        
        yield rag
        
        reset_rag_system()

except ImportError:
    # rag_system.py doesn't exist yet - skip these fixtures
    pass