"""
Test Suite for FlowEdit FastAPI Endpoints and Web UI.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.api.main import app, get_preset_voices

client = TestClient(app)


def test_root_serves_html():
    """Verify GET / and GET /ui serve the FlowEdit HTML UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "FlowEdit" in response.text

    response_ui = client.get("/ui")
    assert response_ui.status_code == 200
    assert "text/html" in response_ui.headers.get("content-type", "")


def test_api_status_endpoint():
    """Verify GET /api/status returns valid JSON structure."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_api_speakers_endpoint():
    """Verify GET /api/speakers returns list of available voices."""
    response = client.get("/api/speakers")
    assert response.status_code == 200
    data = response.json()
    assert "speakers" in data
    assert isinstance(data["speakers"], list)


def test_get_preset_voices():
    """Verify preset voice discovery finds audio files."""
    voices = get_preset_voices()
    print(f"Discovered preset voices: {list(voices.keys())}")
    assert isinstance(voices, dict)


if __name__ == "__main__":
    test_root_serves_html()
    test_api_status_endpoint()
    test_api_speakers_endpoint()
    test_get_preset_voices()
    print("\n[SUCCESS] ALL API AND UI TESTS PASSED!")
