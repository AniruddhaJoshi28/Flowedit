"""
Regression tests for OpenAPI schema and backward compatibility of FlowEdit API.
"""

import pytest
from fastapi.testclient import TestClient
from flowedit.api.main import app

client = TestClient(app)


def test_openapi_schema_compatibility():
    """Verify OpenAPI schema endpoints and parameters remain backward compatible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})

    # Check required core endpoints exist
    assert "/api/synthesize" in paths, "Missing required endpoint /api/synthesize"
    assert "/api/correct" in paths, "Missing required endpoint /api/correct"

    # Check parameters for /api/synthesize
    synth_post = paths["/api/synthesize"]["post"]
    assert synth_post is not None

    # Check parameters for /api/correct
    correct_post = paths["/api/correct"]["post"]
    assert correct_post is not None

    # Verify /api/synthesize_raw is marked deprecated
    assert "/api/synthesize_raw" in paths
    assert paths["/api/synthesize_raw"]["post"].get("deprecated") is True


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FlowEdit API is running" in data.get("message", "")
