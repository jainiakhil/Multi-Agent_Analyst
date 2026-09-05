"""Unit and API tests for dynamic LLM model switching and LLMFactory."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.llm_factory import LLMFactory
from langchain_ollama import ChatOllama

client = TestClient(app)


def test_llm_factory_initial_state():
    """Validates that LLMFactory initializes with application settings defaults."""
    factory = LLMFactory()
    active = factory.get_active_models()
    assert "supervisor_model" in active
    assert "code_analyst_model" in active
    assert "doc_parser_model" in active
    assert active["provider"] == "ollama"


def test_llm_factory_switch_models():
    """Validates that switch_models updates the active models in memory."""
    factory = LLMFactory()
    updated = factory.switch_models(
        supervisor_model="llama3.2:3b",
        code_analyst_model="codellama:7b",
        provider="openai_compatible",
    )
    assert updated["supervisor_model"] == "llama3.2:3b"
    assert updated["code_analyst_model"] == "codellama:7b"
    assert updated["provider"] == "openai_compatible"


def test_llm_factory_create_ollama_chat_model():
    """Validates that create_chat_model returns a ChatOllama instance by default."""
    factory = LLMFactory()
    model = factory.create_chat_model(model_name="test-model:latest", provider="ollama")
    assert isinstance(model, ChatOllama)
    assert model.model == "test-model:latest"


def test_llm_factory_available_models_mock():
    """Validates that get_available_ollama_models parses Ollama tags API response."""
    factory = LLMFactory()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "qwen2.5-coder:7b"},
            {"name": "nomic-embed-text"},
        ]
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        models = factory.get_available_ollama_models()
        assert models == ["llama3.1:8b", "qwen2.5-coder:7b", "nomic-embed-text"]


def test_api_list_models():
    """Verifies that GET /api/v1/models returns active models and available Ollama models."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_models" in data
    assert "available_ollama_models" in data
    assert "supervisor_model" in data["active_models"]


def test_api_switch_models_endpoint():
    """Verifies that POST /api/v1/models/switch dynamically updates the active models."""
    payload = {
        "supervisor_model": "mistral:7b",
        "code_analyst_model": "qwen2.5-coder:14b",
    }
    response = client.post("/api/v1/models/switch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["active_models"]["supervisor_model"] == "mistral:7b"
    assert data["active_models"]["code_analyst_model"] == "qwen2.5-coder:14b"

    # Reset back to defaults
    client.post(
        "/api/v1/models/switch",
        json={
            "supervisor_model": "llama3.1:8b",
            "code_analyst_model": "qwen2.5-coder:7b",
        },
    )
