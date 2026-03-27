# -*- coding: utf-8 -*-
"""Tests for OpenAICompatibleJsonClient backed by OpenAI SDK."""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from agentic_planner.generator.http_llm import OpenAICompatibleJsonClient


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str, model: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        self.model = model


class _FakeCompletions:
    def __init__(self, model: str) -> None:
        self._model = model
        self.last_kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Dict[str, Any]) -> _FakeResponse:
        self.last_kwargs = dict(kwargs)
        if kwargs.get("response_format"):
            return _FakeResponse('{"ok": true}', self._model)
        return _FakeResponse("plain text", self._model)


class _FakeChat:
    def __init__(self, model: str) -> None:
        self.completions = _FakeCompletions(model)


class _FakeOpenAIClient:
    def __init__(self, model: str) -> None:
        self.chat = _FakeChat(model)


def test_complete_json_uses_openai_client_and_captures_usage() -> None:
    client = OpenAICompatibleJsonClient(
        model="fake-model",
        api_key="x",
        base_url="https://example.com/v1",
        openai_client=_FakeOpenAIClient("fake-model"),
    )

    result = client.complete_json("sys", "user")

    assert result == {"ok": True}
    usage = client.get_last_usage()
    assert usage["model"] == "fake-model"
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_generate_uses_openai_client_and_updates_usage() -> None:
    fake = _FakeOpenAIClient("fake-model")
    client = OpenAICompatibleJsonClient(
        model="fake-model",
        api_key="x",
        base_url="https://example.com/v1",
        openai_client=fake,
        default_chat_params={"extra_body": {"enable_thinking": False}},
    )

    text = client.generate("sys", "user", 0.2)

    assert text == "plain text"
    usage = client.get_last_usage()
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert fake.chat.completions.last_kwargs.get("extra_body") == {"enable_thinking": False}


def test_create_client_raises_clear_error_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = __import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("no module named openai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked_import)

    with pytest.raises(ImportError):
        OpenAICompatibleJsonClient(
            model="fake-model",
            api_key="x",
            base_url="https://example.com/v1",
        )
