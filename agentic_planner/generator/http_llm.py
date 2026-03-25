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

import httpx

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
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or ""
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
        self._own_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_sec)

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatibleJsonClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        if not self._api_key:
            raise ValueError(
                "api_key is required (pass api_key=... or set OPENAI_API_KEY in the environment).",
            )
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self._use_json_object:
            body["response_format"] = {"type": "json_object"}

        resp = self._client.post(url, headers=headers, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected chat completions response shape: {data!r}") from exc
        if content is None or (isinstance(content, str) and not content.strip()):
            raise ValueError("empty assistant message content")
        text = content if isinstance(content, str) else str(content)
        return parse_json_object_strict(text)