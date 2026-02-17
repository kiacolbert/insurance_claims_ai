"""Pytest configuration and fixtures"""

import pytest
import os
from pathlib import Path
import shutil
import chromadb
from insurance_claims_ai.rag_system import RAGSystem, MockLLMClient, reset_rag_system


# Set test environment variables BEFORE any imports
os.environ["TESTING"] = "true"
os.environ["CHROMA_DB_PATH"] = str(Path(__file__).parent.parent / "test_chroma_db")
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"


@pytest.fixture(scope="session")
def test_chroma_path(tmp_path_factory):
    """Create temporary ChromaDB directory for tests"""
    test_db = tmp_path_factory.mktemp("test_chroma_db")
    os.environ["CHROMA_DB_PATH"] = str(test_db)
    
    yield test_db
    
    # Cleanup after all tests
    shutil.rmtree(test_db, ignore_errors=True)


@pytest.fixture(scope="session")
def populate_test_db(test_chroma_path):
    """Populate test database with sample documents"""
    client = chromadb.PersistentClient(path=str(test_chroma_path))
    
    try:
        collection = client.get_collection("insurance_docs")
    except:
        collection = client.create_collection("insurance_docs")
    
    # Add test documents
    collection.add(
        documents=[
            "Your collision deductible is $500 per incident.",
            "To file a claim, call 1-800-CLAIMS or visit our website within 24 hours of the incident.",
            "Liability coverage limit is $100,000 per person, $300,000 per accident.",
            "Comprehensive coverage includes theft, vandalism, fire, and weather damage.",
            "Rental car reimbursement covers up to $30 per day for a maximum of 30 days.",
            "Roadside assistance is available 24/7 at no additional charge.",
            "Your policy covers medical payments up to $5,000 per person.",
            "Uninsured motorist coverage provides protection up to $50,000 per person."
        ],
        ids=[f"doc{i}" for i in range(1, 9)],
        metadatas=[
            {"source": "policy.pdf", "page": 1, "section": "Deductibles"},
            {"source": "claims_guide.pdf", "page": 1, "section": "Filing"},
            {"source": "policy.pdf", "page": 2, "section": "Liability"},
            {"source": "policy.pdf", "page": 3, "section": "Comprehensive"},
            {"source": "policy.pdf", "page": 4, "section": "Rental"},
            {"source": "policy.pdf", "page": 5, "section": "Roadside"},
            {"source": "policy.pdf", "page": 6, "section": "Medical"},
            {"source": "policy.pdf", "page": 7, "section": "Uninsured"}
        ]
    )
    
    yield collection


@pytest.fixture(scope="function")
def mock_rag_system(test_chroma_path, populate_test_db):
    """Create RAG system with mock LLM for each test"""
    # Reset singleton before each test
    reset_rag_system()
    
    # Create RAG system with mock client
    rag = RAGSystem(
        chroma_path=test_chroma_path,
        llm_client=MockLLMClient()
    )
    
    yield rag
    
    # Reset after test
    reset_rag_system()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from insurance_claims_ai.api import app, get_rag_function, MockRAGFunction
    
    # Override the dependency with mock
    app.dependency_overrides[get_rag_function] = lambda: MockRAGFunction()
    
    with TestClient(app) as c:
        yield c
    
    # Clean up overrides after tests
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_cache(client):
    """Clear cache before each test"""
    client.delete("/cache")
    yield