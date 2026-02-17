# tests/conftest.py
import pytest
import os
from pathlib import Path
from unittest.mock import patch

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
    return MOCK_CONTEXTS[:top_k]


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import insurance_claims_ai.api as api_module

    # Mock retrieve_context - this is what's hitting the real DB
    with patch.object(api_module, "retrieve_context", mock_retrieve_context):
        with TestClient(api_module.app) as c:
            yield c

    api_module.app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_cache(client):
    client.delete("/cache")
    yield