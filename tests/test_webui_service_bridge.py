# -*- coding: utf-8 -*-
"""Tests for ServiceBridge test_llm functionality."""

from unittest.mock import MagicMock, patch


def test_test_llm_returns_success_with_valid_config():
    """Test that test_llm uses generate() method correctly."""
    from apps.webui.backend.service_bridge import ServiceBridge

    bridge = ServiceBridge()

    with patch("apps.webui.backend.service_bridge.OpenAICompatibleJsonClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate.return_value = "ok"
        mock_client_class.return_value = mock_client

        result = bridge.test_llm(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        assert result.ok is True
        assert result.data["model"] == "gpt-4"
        assert result.data["message"] == "LLM connection successful"
        assert "response_preview" in result.data
        mock_client.generate.assert_called_once_with(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with 'ok' only.",
        )
        mock_client.close.assert_called_once()


def test_test_llm_returns_error_without_model():
    """Test that test_llm returns error when model is missing."""
    from apps.webui.backend.service_bridge import ServiceBridge

    bridge = ServiceBridge()
    result = bridge.test_llm(base_url="https://api.openai.com/v1", api_key="test-key", model="")

    assert result.ok is False
    assert "Model name is required" in result.error


def test_test_llm_returns_error_without_api_key():
    """Test that test_llm returns error when API key is missing."""
    from apps.webui.backend.service_bridge import ServiceBridge

    bridge = ServiceBridge()
    result = bridge.test_llm(
        base_url="https://api.openai.com/v1",
        api_key="",
        model="gpt-4",
    )

    assert result.ok is False
    assert "API key is required" in result.error


def test_test_llm_handles_exception():
    """Test that test_llm handles exceptions gracefully."""
    from apps.webui.backend.service_bridge import ServiceBridge

    bridge = ServiceBridge()

    with patch("apps.webui.backend.service_bridge.OpenAICompatibleJsonClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        result = bridge.test_llm(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        assert result.ok is False
        assert "LLM connection failed" in result.error
        assert "Connection failed" in result.error
