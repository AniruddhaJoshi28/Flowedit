"""
Regression tests for OpenAPI schema of FlowEdit API.
"""

import pytest
from fastapi.testclient import TestClient
from flowedit.api.main import app

client = TestClient(app)


def test_openapi_schema_compatibility():
    """Verify OpenAPI schema endpoints and parameters."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})

    # Check required core endpoints exist
    assert "/api/synthesize" in paths, "Missing required endpoint /api/synthesize"
    assert "/api/correct" in paths, "Missing required endpoint /api/correct"
    assert "/api/baseline" in paths, "Missing required endpoint /api/baseline"
    assert "/api/memory" in paths, "Missing required endpoint /api/memory"

    synth_post = paths["/api/synthesize"]["post"]
    assert synth_post is not None

    correct_post = paths["/api/correct"]["post"]
    assert correct_post is not None


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FlowEdit API" in data.get("message", "")


def test_memory_endpoint_structure():
    """Verify /api/memory endpoint returns correct schema."""
    response = client.get("/api/memory")
    assert response.status_code == 200
    data = response.json()
    assert "corrections" in data
    assert "size" in data
    assert isinstance(data["corrections"], list)
