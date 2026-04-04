# -*- coding: utf-8 -*-
"""FastAPI route definitions for WebUI."""

from __future__ import annotations

from fastapi import APIRouter

from apps.webui.backend.schemas import (
    DispatchRequest,
    ExplainRequest,
    GenerateRequest,
    OptimizeRequest,
    ToolResponseModel,
    ValidateRequest,
)
from apps.webui.backend.service_bridge import ServiceBridge

router = APIRouter()
bridge = ServiceBridge()


@router.post("/dispatch", response_model=ToolResponseModel)
def dispatch(req: DispatchRequest) -> ToolResponseModel:
    resp = bridge.dispatch(req.action, req.payload)
    return ToolResponseModel(**resp.__dict__)


@router.post("/generate", response_model=ToolResponseModel)
def generate(req: GenerateRequest) -> ToolResponseModel:
    payload = {
        "intent": req.intent,
        "dataset_path": req.dataset_path,
        "model_config_path": req.model_config_path,
        "options": req.options,
    }
    resp = bridge.dispatch("generate", payload)
    return ToolResponseModel(**resp.__dict__)


@router.post("/optimize", response_model=ToolResponseModel)
def optimize(req: OptimizeRequest) -> ToolResponseModel:
    payload = {
        "yaml_text_or_path": req.yaml_text_or_path,
        "objective": req.objective,
        "model_config_path": req.model_config_path,
        "options": req.options,
    }
    resp = bridge.dispatch("optimize", payload)
    return ToolResponseModel(**resp.__dict__)


@router.post("/validate", response_model=ToolResponseModel)
def validate(req: ValidateRequest) -> ToolResponseModel:
    payload = {
        "yaml_text_or_path": req.yaml_text_or_path,
        "options": req.options,
    }
    resp = bridge.dispatch("validate", payload)
    return ToolResponseModel(**resp.__dict__)


@router.post("/list_ops", response_model=ToolResponseModel)
def list_ops() -> ToolResponseModel:
    resp = bridge.dispatch("list_ops", {})
    return ToolResponseModel(**resp.__dict__)


@router.post("/explain_op", response_model=ToolResponseModel)
def explain_op(req: ExplainRequest) -> ToolResponseModel:
    payload = {
        "operator_name": req.operator_name,
        "options": req.options,
    }
    resp = bridge.dispatch("explain_op", payload)
    return ToolResponseModel(**resp.__dict__)
