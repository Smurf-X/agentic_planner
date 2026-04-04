# -*- coding: utf-8 -*-
"""Tests for WebUI backend service bridge."""


def test_service_bridge_dispatch_returns_tool_response():
    from apps.webui.backend.service_bridge import ServiceBridge

    bridge = ServiceBridge()
    resp = bridge.dispatch("list_ops", {})
    assert resp.ok is True
    assert "operators" in resp.data