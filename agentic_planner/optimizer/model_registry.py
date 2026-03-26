# -*- coding: utf-8 -*-
"""
Model registry for managing LLM model configurations.

This module provides:
1. ModelConfig - Configuration for a single LLM model
2. JudgeConfig - Configuration for LLM-as-Judge
3. ModelRegistry - Central registry for all models used in optimization
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    model: str = Field(..., description="Model identifier (e.g., 'gpt-4o-mini')")
    api_key: str = Field(default="", description="API key for this model")
    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    price_per_million: float = Field(default=0.0, description="Price per million tokens (USD)")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    supports_json_mode: bool = Field(default=True, description="Whether model supports JSON mode")

    model_config = {"extra": "allow"}


class JudgeConfig(BaseModel):
    """Configuration for LLM-as-Judge."""

    model: str = Field(default="gpt-4o-mini", description="Judge model identifier")
    api_key: str = Field(default="", description="API key for judge")
    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    price_per_million: float = Field(default=0.0, description="Price per million tokens for judge")
    temperature: float = Field(default=0.1, description="Temperature for judge calls")

    model_config = {"extra": "allow"}


class ModelsConfig(BaseModel):
    """Full model configuration for optimization."""

    judge: JudgeConfig = Field(default_factory=JudgeConfig, description="Judge configuration")
    pipeline_models: Dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Available models for pipeline operators",
    )
    candidate_models: List[str] = Field(
        default_factory=list,
        description="Models to try during optimization (subset of pipeline_models keys)",
    )

    model_config = {"extra": "allow"}

    @classmethod
    def from_yaml(cls, path: str) -> "ModelsConfig":
        """Load configuration from YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()
        with p.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


class ModelRegistry:
    """
    Central registry for managing LLM model configurations.

    This class provides:
    - Model configuration lookup by name
    - Price calculation for cost estimation
    - API client creation for each model
    - Candidate model list for optimization

    Example:
        registry = ModelRegistry.from_yaml("models.yaml")

        # Get model config
        config = registry.get_model("gpt-4o-mini")

        # Get price
        price = registry.get_price("gpt-4o-mini")

        # Create LLM client
        client = registry.create_client("gpt-4o-mini")
    """

    def __init__(self, config: ModelsConfig) -> None:
        """
        Initialize the registry.

        Args:
            config: ModelsConfig containing all model configurations
        """
        self._config = config
        self._model_configs: Dict[str, ModelConfig] = {}

        for name, cfg in config.pipeline_models.items():
            if isinstance(cfg, dict):
                cfg = ModelConfig(**cfg)
            self._model_configs[name] = cfg

    @classmethod
    def from_yaml(cls, path: str) -> "ModelRegistry":
        """Create registry from YAML configuration file."""
        config = ModelsConfig.from_yaml(path)
        return cls(config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRegistry":
        """Create registry from dictionary."""
        config = ModelsConfig.model_validate(data)
        return cls(config)

    @classmethod
    def default(cls) -> "ModelRegistry":
        """Create registry with default configuration."""
        default_config = {
            "judge": {
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "price_per_million": 0.15,
            },
            "pipeline_models": {
                "gpt-4o-mini": {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "price_per_million": 0.15,
                },
                "gpt-4o": {
                    "model": "gpt-4o",
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "price_per_million": 2.5,
                },
            },
            "candidate_models": ["gpt-4o-mini", "gpt-4o"],
        }
        return cls.from_dict(default_config)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._model_configs.get(name)

    def get_model_or_raise(self, name: str) -> ModelConfig:
        """Get model configuration or raise error if not found."""
        cfg = self.get_model(name)
        if cfg is None:
            raise ValueError(
                f"Model '{name}' not found in registry. Available: {list(self._model_configs.keys())}"
            )
        return cfg

    def get_price(self, name: str) -> float:
        """Get price per million tokens for a model."""
        cfg = self.get_model(name)
        if cfg:
            return cfg.price_per_million
        return 0.0

    def get_price_table(self) -> Dict[str, float]:
        """Get price table for all models."""
        return {name: cfg.price_per_million for name, cfg in self._model_configs.items()}

    def get_candidate_models(self) -> List[str]:
        """Get list of candidate models for optimization."""
        if self._config.candidate_models:
            return [name for name in self._config.candidate_models if name in self._model_configs]
        return list(self._model_configs.keys())

    def has_model(self, name: str) -> bool:
        """Return whether the registry contains a model."""
        return name in self._model_configs

    def is_swap_compatible(self, from_model: str, to_model: str) -> bool:
        """Check whether swapping between two models is considered safe."""
        if not to_model:
            return False
        if from_model == to_model:
            return False

        if not self.has_model(to_model):
            return False

        to_cfg = self.get_model_or_raise(to_model)
        if not to_cfg.supports_json_mode:
            return False

        if not from_model:
            return True
        if not self.has_model(from_model):
            return False

        from_cfg = self.get_model_or_raise(from_model)
        if from_cfg.base_url != to_cfg.base_url:
            return False

        return True

    def get_swap_candidates(self, from_model: str) -> List[str]:
        """Return candidate models that are swap-compatible with the current model."""
        return [
            candidate
            for candidate in self.get_candidate_models()
            if self.is_swap_compatible(from_model=from_model, to_model=candidate)
        ]

    def get_judge_config(self) -> JudgeConfig:
        """Get judge configuration."""
        return self._config.judge

    def get_judge_price(self) -> float:
        """Get price per million tokens for judge."""
        return self._config.judge.price_per_million

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._model_configs.keys())

    def create_client(self, name: str):
        """
        Create LLM client for a model.

        Args:
            name: Model name

        Returns:
            OpenAICompatibleJsonClient instance
        """
        from agentic_planner.generator.http_llm import OpenAICompatibleJsonClient

        cfg = self.get_model_or_raise(name)

        return OpenAICompatibleJsonClient(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

    def create_judge_client(self):
        """Create LLM client for judge."""
        from agentic_planner.generator.http_llm import OpenAICompatibleJsonClient

        judge = self._config.judge

        return OpenAICompatibleJsonClient(
            model=judge.model,
            api_key=judge.api_key,
            base_url=judge.base_url,
        )

    def compute_token_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Compute cost for a single model's token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        price = self.get_price(model)
        total_tokens = prompt_tokens + completion_tokens
        return total_tokens / 1_000_000.0 * price

    def compute_total_cost(
        self,
        model_usage: Dict[str, Dict[str, int]],
    ) -> float:
        """
        Compute total cost across all models.

        Args:
            model_usage: Dict mapping model name to {"prompt": int, "completion": int}

        Returns:
            Total cost in USD
        """
        total = 0.0
        for model, usage in model_usage.items():
            prompt = usage.get("prompt", 0)
            completion = usage.get("completion", 0)
            total += self.compute_token_cost(model, prompt, completion)
        return total

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(models={self.list_models()}, candidates={self.get_candidate_models()})"
        )


# Default price table for common models
DEFAULT_PRICES = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 5.0,
    "gpt-4-turbo": 10.0,
    "gpt-3.5-turbo": 0.5,
    "claude-3-haiku": 0.25,
    "claude-3-sonnet": 3.0,
    "claude-3-opus": 15.0,
    "deepseek-chat": 0.1,
    "deepseek-coder": 0.1,
    "qwen-turbo": 0.02,
    "qwen-plus": 0.04,
    "qwen-max": 0.12,
}


__all__ = [
    "ModelConfig",
    "JudgeConfig",
    "ModelsConfig",
    "ModelRegistry",
    "DEFAULT_PRICES",
]
