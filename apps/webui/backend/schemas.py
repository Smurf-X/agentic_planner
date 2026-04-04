# -*- coding: utf-8 -*-
"""Pydantic request/response models for WebUI API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class DispatchRequest(BaseModel):
    action: str
    payload: Dict[str, Any] = {}


class GenerateRequest(BaseModel):
    intent: str
    dataset_path: str
    model_config_path: str
    options: Dict[str, Any] = {}


class OptimizeRequest(BaseModel):
    yaml_text_or_path: str
    objective: str
    model_config_path: str
    options: Dict[str, Any] = {}


class ValidateRequest(BaseModel):
    yaml_text_or_path: str
    options: Dict[str, Any] = {}


class ExplainRequest(BaseModel):
    operator_name: str
    options: Dict[str, Any] = {}


class ToolResponseModel(BaseModel):
    ok: bool
    data: Dict[str, Any] = {}
    timing_ms: int = 1
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
