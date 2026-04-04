# -*- coding: utf-8 -*-
"""Tests for WebUI backend API routes."""


def test_list_ops_route_returns_json():
    from fastapi.testclient import TestClient

    from apps.webui.backend.main import app

    client = TestClient(app)
    resp = client.post("/api/list_ops", json={})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True