# -*- coding: utf-8 -*-
"""
Minimal HTTP client for OpenAI-compatible Chat Completions APIs.

Works with OpenAI, Azure OpenAI (set ``base_url`` + key), DashScope compatible mode,
vLLM ``/v1/chat/completions``, and other providers that accept the same JSON shape.

Environment variables (optional defaults):
- ``OPENAI_API_KEY`` — used when ``api_key`` is omitted.
- ``DASHSCOPE_API_KEY`` — alias for ``OPENAI_API_KEY`` (DashScope / 阿里云兼容模式).
- ``OPENAI_BASE_URL`` — used when ``base_url`` is omitted (default ``https://api.openai.com/v1``).
- ``DASHSCOPE_BASE_URL`` — alias for ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from agentic_planner.generator.llm import parse_json_object_strict


class OpenAICompatibleJsonClient:
    """
    Call ``POST {base_url}/chat/completions`` and parse the assistant message as JSON.

    The model is instructed to return **only** a JSON object; content is parsed with
    :func:`parse_json_object_strict`. If the API supports ``response_format`` JSON mode,
    set ``use_response_json_object=True`` (default) for stricter JSON outputs.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
        use_response_json_object: bool = True,
        openai_client: Optional[Any] = None,
        default_chat_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self._api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
        ).strip() or None
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self._timeout = timeout_sec
        self._extra_headers = dict(extra_headers or {})
        self._use_json_object = use_response_json_object
        self._default_chat_params = dict(default_chat_params or {})
        self._last_usage: Dict[str, Any] = {
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._client = openai_client or self._create_openai_client()

    def _create_openai_client(self) -> Any:
        """Create OpenAI SDK client with OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAICompatibleJsonClient; "
                "install with `pip install openai`"
            ) from exc

        kwargs: Dict[str, Any] = {
            "api_key": self._api_key,
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        if self._extra_headers:
            kwargs["default_headers"] = self._extra_headers
        return OpenAI(**kwargs)

    def _capture_usage(self, response: Any) -> None:
        """Capture token usage from OpenAI-compatible response."""
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", self.model)

        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
            model = response.get("model", self.model)

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if usage is not None:
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)
            else:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        self._last_usage = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def get_last_usage(self) -> Dict[str, Any]:
        """Get token usage from the last request."""
        return dict(self._last_usage)

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "OpenAICompatibleJsonClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        if not self._api_key:
            raise ValueError(
                "api_key is required (pass api_key=... or set OPENAI_API_KEY in the environment).",
            )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        for key, value in self._default_chat_params.items():
            if key not in {"model", "messages"}:
                body[key] = value
        if self._use_json_object:
            body["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**body)
        self._capture_usage(response)
        try:
            content = response.choices[0].message.content
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "unexpected chat completions response shape from OpenAI-compatible client"
            ) from exc
        if content is None or (isinstance(content, str) and not content.strip()):
            raise ValueError("empty assistant message content")
        text = content if isinstance(content, str) else str(content)
        return parse_json_object_strict(text)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text response (non-JSON mode).

        This method is used by LLMActionSelector for action selection.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        if not self._api_key:
            raise ValueError(
                "api_key is required (pass api_key=... or set OPENAI_API_KEY in the environment).",
            )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        for key, value in self._default_chat_params.items():
            if key not in {"model", "messages", "temperature"}:
                body[key] = value

        response = self._client.chat.completions.create(**body)
        self._capture_usage(response)
        try:
            content = response.choices[0].message.content
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "unexpected chat completions response shape from OpenAI-compatible client"
            ) from exc
        return content if isinstance(content, str) else str(content)
