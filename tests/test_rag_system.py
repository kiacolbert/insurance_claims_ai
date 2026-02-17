"""Unit tests for RAG system"""

import pytest
from insurance_claims_ai.rag_system import RAGSystem, MockLLMClient, AnthropicClient


def test_rag_system_initialization(test_chroma_path, populate_test_db):
    """Test RAG system initializes correctly"""
    rag = RAGSystem(
        chroma_path=test_chroma_path,
        llm_client=MockLLMClient()
    )
    
    assert rag.collection is not None
    assert rag.llm_client is not None


def test_rag_query_returns_response(mock_rag_system):
    """Test RAG query returns structured response"""
    result = mock_rag_system.query("What is my deductible?")
    
    assert result.answer is not None
    assert isinstance(result.sources, list)
    assert isinstance(result.context, list)
    assert result.cached is False


def test_rag_retrieves_relevant_docs(mock_rag_system):
    """Test that RAG retrieves relevant documents"""
    result = mock_rag_system.query("What is my deductible?", top_k=3)
    
    # Should retrieve documents about deductibles
    assert len(result.context) > 0
    assert any("deductible" in doc.lower() for doc in result.context)


def test_mock_llm_client():
    """Test mock LLM client returns expected responses"""
    client = MockLLMClient()
    
    # Test deductible question
    response = client.generate("What is my deductible?")
    assert "$500" in response
    
    # Test claim question
    response = client.generate("How do I file a claim?")
    assert "1-800-CLAIMS" in response


def test_rag_with_different_top_k(mock_rag_system):
    """Test RAG with different top_k values"""
    result_k1 = mock_rag_system.query("What is covered?", top_k=1)
    result_k5 = mock_rag_system.query("What is covered?", top_k=5)
    
    assert len(result_k1.context) <= 1
    assert len(result_k5.context) <= 5


@pytest.mark.integration
def test_real_anthropic_client():
    """Integration test with real Anthropic API (skipped by default)"""
    import os
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "test-key-not-real":
        pytest.skip("Real API key not available")
    
    from insurance_claims_ai.rag_system import AnthropicClient
    client = AnthropicClient(api_key)
    response = client.generate("Say 'Hello, World!' and nothing else.")
    
    assert "Hello" in response
    
    