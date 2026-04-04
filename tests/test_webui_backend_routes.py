# -*- coding: utf-8 -*-
"""Tests for WebUI backend API routes."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_list_ops_route_returns_json():
    from apps.webui.backend.main import app

    client = TestClient(app)
    resp = client.post("/api/list_ops", json={})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_generate_handles_llm_error_gracefully():
    """Test that generate returns proper error when LLM fails."""
    from apps.webui.backend.main import app
    from agent_runtime.api.schemas import ToolResponse

    client = TestClient(app)

    with patch("apps.webui.backend.service_bridge.ServiceBridge.dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResponse(
            ok=False,
            error="LLM API error: rate limit exceeded",
            data={},
            timing_ms=0,
        )

        resp = client.post(
            "/api/generate",
            json={
                "intent": "test",
                "dataset_path": "/path/to/data.jsonl",
                "llm_config": {"model": "gpt-4o-mini", "api_key": "test"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data
        assert "rate limit" in data["error"].lower()


def test_generate_handles_llm_invalid_json_response():
    """Test that generate returns proper error when LLM returns non-JSON."""
    from apps.webui.backend.main import app
    from agent_runtime.api.schemas import ToolResponse

    client = TestClient(app)

    with patch("apps.webui.backend.service_bridge.ServiceBridge.dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResponse(
            ok=False,
            error="generation failed: LLM returned invalid JSON: Unexpected token 'I' at position 0",
            data={},
            timing_ms=0,
        )

        resp = client.post(
            "/api/generate",
            json={
                "intent": "test",
                "dataset_path": "/path/to/data.jsonl",
                "llm_config": {"model": "gpt-4o-mini", "api_key": "test"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data
        assert "invalid JSON" in data["error"]