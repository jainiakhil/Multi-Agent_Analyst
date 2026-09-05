"""Integration tests for FastAPI REST and WebSocket endpoints."""

import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoints():
    """Verifies that the /health and /api/v1/health endpoints return status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "ollama_base_url" in data

    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 200
    assert response_v1.json()["status"] == "healthy"


def test_document_upload_invalid_extension():
    """Verifies that uploading an unsupported file type returns a 400 Bad Request."""
    file_content = b"Some binary or unsupported content"
    files = {"file": ("malicious.exe", io.BytesIO(file_content), "application/octet-stream")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_websocket_endpoint_missing_query():
    """Verifies that connecting and sending an empty/invalid payload triggers an error message."""
    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json({})
        data = websocket.receive_json()
        assert "error" in data
        assert "Missing required field 'query'" in data["error"]
