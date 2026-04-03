# -*- coding: utf-8 -*-
"""LLM-driven directive inference engine.

Analyzes a pipeline configuration and recommends which directives to apply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import (
    DIRECTIVE_REGISTRY,
    get_directive_spec,
    list_directive_names,
)
from agentic_planner.optimizer.op_locator import ProcessIndex, TargetLocator

if TYPE_CHECKING:
    from agentic_planner.generator.llm import BaseLLMClient


class DirectiveRecommendation(BaseModel):
    """A single recommended directive with rationale."""

    directive_name: str = Field(description="Name of the directive to apply")
    directive_template: str = Field(
        default="",
        description="Directive spec/template name (preferred over directive_name)",
    )
    target_locator: Optional[Dict[str, str]] = Field(
        default=None,
        description="Canonical target locator payload for per-operator directives",
    )
    instantiate_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime params for template instantiation",
    )
    priority: int = Field(default=0, ge=0, le=10, description="Priority (higher = apply earlier)")
    rationale: str = Field(default="", description="Why this directive is recommended")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy alias of instantiate_params (kept for compatibility)",
    )

    def resolved_template(self) -> str:
        """Resolve directive template name with legacy fallbacks."""
        if self.directive_template:
            return self.directive_template

        alias_map = {
            "tighten_filters": "tighten_threshold",
            "loosen_filters": "loosen_threshold",
            "remove_redundant_ops": "remove_redundant_op",
            "reorder_filters_first": "safe_reorder_local",
        }
        return alias_map.get(self.directive_name, self.directive_name)

    def resolved_params(self) -> Dict[str, Any]:
        """Resolve instantiate params with backward-compatible aliasing."""
        if self.instantiate_params:
            return dict(self.instantiate_params)
        return dict(self.params)

    def to_target_locator(self) -> Optional[TargetLocator]:
        """Parse target locator payload into typed locator."""
        payload = self.target_locator
        if payload is None:
            return None

        operator_id = payload.get("operator_id")
        audit_identity_hash = payload.get("audit_identity_hash")
        if not operator_id or not audit_identity_hash:
            return None

        return TargetLocator(
            operator_id=operator_id,
            audit_identity_hash=audit_identity_hash,
        )


class DirectiveRecommendations(BaseModel):
    """Result of LLM-based directive analysis."""

    recommendations: List[DirectiveRecommendation] = Field(default_factory=list)
    analysis_summary: str = Field(default="", description="Brief summary of pipeline analysis")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass
class InferredDirectiveEngineRun:
    """Result of running the inferred directive engine."""

    ok: bool
    config: DJExecutableConfig
    recommendations: DirectiveRecommendations
    trace: List[DirectiveResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DirectiveInferencePromptBuilder:
    """Builds prompts for directive inference."""

    INFERENCE_SYSTEM_PROMPT = """You are a data pipeline optimization expert. Analyze the given Data-Juicer pipeline configuration and recommend which optimization directives to apply.

Available directives:
{directive_descriptions}

Your task:
1. Analyze the pipeline structure and operators
2. Identify optimization opportunities
3. Recommend specific directives in priority order
4. Explain your reasoning for each recommendation

Guidelines:
- Focus on practical, high-impact optimizations
- Consider trade-offs between cost and quality
- Prioritize simple, safe transformations
- Be specific about parameters when needed"""

    INFERENCE_USER_PROMPT = """Analyze this Data-Juicer pipeline configuration and recommend optimizations:

```yaml
{config_yaml}
```

Task context: {task_description}

Optimization goal: {optimization_goal}

Return your recommendations as a JSON object with this structure:
{
  "analysis_summary": "Brief summary of pipeline analysis",
  "confidence": 0.0-1.0,
  "recommendations": [
    {
      "directive_name": "name of directive",
      "directive_template": "directive spec/template key",
      "target_locator": {"operator_id": "op-123", "audit_identity_hash": "hash"},
      "instantiate_params": {},
      "priority": 0-10,
      "rationale": "why this helps",
      "params": {}
    }
  ]
}"""

    @classmethod
    def build_prompt(
        cls,
        config: DJExecutableConfig,
        task_description: str = "",
        optimization_goal: str = "balance cost and quality",
    ) -> tuple[str, str]:
        """Build system and user prompts for inference."""
        import yaml

        # Build directive descriptions
        directive_desc = cls._build_directive_descriptions()

        # Convert config to YAML
        config_yaml = yaml.safe_dump(config, allow_unicode=True, default_flow_style=False)

        system_prompt = cls.INFERENCE_SYSTEM_PROMPT.format(
            directive_descriptions=directive_desc
        )
        user_prompt = cls.INFERENCE_USER_PROMPT.format(
            config_yaml=config_yaml,
            task_description=task_description or "general data processing",
            optimization_goal=optimization_goal,
        )

        return system_prompt, user_prompt

    @classmethod
    def _build_directive_descriptions(cls) -> str:
        """Build descriptions of available directives."""
        descriptions = []
        for name in list_directive_names():
            d = DIRECTIVE_REGISTRY.get(name)
            if d:
                doc = d.__class__.__doc__ or "No description available."
                desc = f"- **{name}**: {doc.strip().split(chr(10))[0]}"
                descriptions.append(desc)
        return "\n".join(descriptions)


class DirectiveInferenceEngine:
    """
    Uses LLM to analyze pipeline and recommend directives.

    This is the "smart" mode of directive-based optimization where
    the LLM reasons about which directives to apply rather than
    using a static list.
    """

    def __init__(
        self,
        llm_client: Optional["BaseLLMClient"] = None,
        task_description: str = "",
        optimization_goal: str = "balance cost and quality",
        max_directives: int = 10,
    ) -> None:
        """
        Args:
            llm_client: LLM client for inference (optional, can be set later)
            task_description: Description of the data processing task
            optimization_goal: What to optimize for (cost, quality, or balance)
            max_directives: Maximum number of directives to recommend
        """
        self._llm_client = llm_client
        self._task_description = task_description
        self._optimization_goal = optimization_goal
        self._max_directives = max_directives

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client."""
        self._llm_client = client

    def analyze(self, cfg: DJExecutableConfig) -> DirectiveRecommendations:
        """
        Analyze the configuration and recommend directives.

        Returns recommendations even if LLM is not available (falls back to heuristics).
        """
        if self._llm_client:
            return self._analyze_with_llm(cfg)
        return self._analyze_with_heuristics(cfg)

    def _analyze_with_llm(self, cfg: DJExecutableConfig) -> DirectiveRecommendations:
        """Use LLM to analyze and recommend."""
        system_prompt, user_prompt = DirectiveInferencePromptBuilder.build_prompt(
            cfg,
            self._task_description,
            self._optimization_goal,
        )

        try:
            response = self._llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
            )
            data = self._parse_json_object(response)
            return DirectiveRecommendations.model_validate(data)
        except Exception:
            # Fall back to heuristics on LLM error
            return self._analyze_with_heuristics(cfg)

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        """Parse JSON object from plain or fenced model output."""
        stripped = text.strip()

        candidates: List[str] = [stripped]
        if "```json" in stripped:
            candidates.append(stripped.split("```json", 1)[1].split("```", 1)[0].strip())
        if "```" in stripped:
            candidates.append(stripped.split("```", 1)[1].split("```", 1)[0].strip())

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(stripped[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data

        raise ValueError("Failed to parse JSON object from model response")

    def _analyze_with_heuristics(self, cfg: DJExecutableConfig) -> DirectiveRecommendations:
        """Simple heuristic-based analysis (fallback)."""
        recommendations: List[DirectiveRecommendation] = []
        proc = cfg.get("process", [])
        index = ProcessIndex.build(proc if isinstance(proc, list) else [])

        # Always recommend reorder if there are multiple operators
        if len(proc) >= 2:
            recommendations.append(
                DirectiveRecommendation(
                    directive_name="reorder_filters_first",
                    directive_template="safe_reorder_local",
                    priority=10,
                    rationale="Move filters before mappers to reduce downstream data volume",
                )
            )

        # Recommend redundant op removal
        recommendations.append(
            DirectiveRecommendation(
                directive_name="remove_redundant_ops",
                directive_template="remove_redundant_op",
                priority=9,
                rationale="Remove duplicate or no-op operators",
            )
        )

        # Check for text_length_filter with low min_len
        for i, step in enumerate(proc):
            if not isinstance(step, dict) or len(step) != 1:
                continue
            name = next(iter(step.keys()))
            params = step.get(name, {})
            if name == "text_length_filter" and isinstance(params, dict):
                min_len = params.get("min_len", 0)
                if min_len < 10:
                    locator = None
                    if i < len(index.identities):
                        locator = index.identities[i].to_target_locator().to_dict()
                    recommendations.append(
                        DirectiveRecommendation(
                            directive_name="bump_text_length_min_len",
                            directive_template="bump_text_length_min_len",
                            target_locator=locator,
                            instantiate_params={"delta": 10},
                            priority=5,
                            rationale=f"Increase min_len from {min_len} to filter more noise",
                        )
                    )

        # Limit recommendations
        recommendations = sorted(recommendations, key=lambda r: -r.priority)[: self._max_directives]

        return DirectiveRecommendations(
            recommendations=recommendations,
            analysis_summary="Heuristic-based analysis (LLM unavailable)",
            confidence=0.6,
        )

    def run(self, cfg: DJExecutableConfig) -> InferredDirectiveEngineRun:
        """
        Analyze the config and apply recommended directives.

        This combines analysis with execution.
        """
        from copy import deepcopy

        recommendations = self.analyze(cfg)
        current = deepcopy(cfg)
        trace: List[DirectiveResult] = []
        errors: List[str] = []

        for rec in recommendations.recommendations:
            template_name = rec.resolved_template()
            runtime_name = rec.directive_name or template_name
            params = rec.resolved_params()
            target_locator = rec.to_target_locator()

            result: Optional[DirectiveResult] = None
            spec = get_directive_spec(template_name)
            if spec is not None:
                instantiated = spec.instantiate(params=params, target_locator=target_locator)
                result = instantiated.apply(current)
                runtime_name = instantiated.directive.name
            else:
                directive = DIRECTIVE_REGISTRY.get(runtime_name) or DIRECTIVE_REGISTRY.get(template_name)
                if directive is None:
                    errors.append(f"Unknown directive: {runtime_name}")
                    continue

                target_index = None
                if target_locator is not None:
                    target_index = ProcessIndex.build(current.get("process", [])).locate_target(target_locator)
                    if target_index is None:
                        errors.append(
                            "target operator no longer exists "
                            f"(operator_id={target_locator.operator_id})"
                        )
                        continue
                result = directive.apply(current, target_op=target_index)

            if result is None:
                errors.append(f"{runtime_name}: failed to execute")
                continue

            trace.append(result)

            if not result.ok:
                errors.append(f"{runtime_name}: {result.message}")
                continue

            if result.applied and result.config_after:
                current = result.config_after

        return InferredDirectiveEngineRun(
            ok=len(errors) == 0,
            config=current,
            recommendations=recommendations,
            trace=trace,
            errors=errors,
        )
