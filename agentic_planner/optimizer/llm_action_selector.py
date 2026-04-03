# -*- coding: utf-8 -*-
"""
LLM-based action selector for pipeline optimization.

This module provides intelligent action selection using LLM to identify
the most promising (operator, directive) combinations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agentic_planner.optimizer.op_locator import TargetLocator

if TYPE_CHECKING:
    from agentic_planner.optimizer.action import Action, ActionSpace
    from agentic_planner.optimizer.action_context import ActionSelectionContext


@dataclass
class ActionSelectionResult:
    """Result from LLM action selection."""

    selected_actions: List[Action]
    """Selected actions."""

    scores: Dict[str, float] = field(default_factory=dict)
    """Action name -> score mapping."""

    reasoning: str = ""
    """LLM's reasoning for selection."""

    llm_cost: float = 0.0
    """Cost of LLM call."""

    planned_selections: List["PlannedActionSelection"] = field(default_factory=list)
    """Structured plans containing template + target + instantiation params."""


@dataclass
class PlannedActionSelection:
    """Structured plan emitted by action selection."""

    directive_template: str
    target_locator: TargetLocator
    instantiate_params: Dict[str, Any] = field(default_factory=dict)
    directive_name: str = ""
    score: float = 1.0
    reason: str = ""


class LLMActionSelector:
    """
    LLM-based selector for choosing promising actions from action space.

    This class uses an LLM to analyze the pipeline state and select
    the most likely-to-improve actions, reducing the search space.

    Example:
        selector = LLMActionSelector(llm_client=client)
        result = selector.select(action_space, context, top_k=10)
        for action in result.selected_actions:
            print(f"Selected: {action}")
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 2,
    ):
        """
        Initialize the selector.

        Args:
            llm_client: LLM client with generate() method
            model: Model name for LLM calls
            temperature: Temperature for generation
            max_retries: Maximum retries on parse failure
        """
        self._llm_client = llm_client
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client."""
        self._llm_client = client

    def select_with_context(
        self,
        action_space: ActionSpace,
        context: ActionSelectionContext,
        top_k: int = 10,
    ) -> ActionSelectionResult:
        """
        Select top-k actions using LLM with rich context.

        This method uses ActionSelectionContext which provides:
        - Execution results sample
        - Data sample
        - Action performance history
        - Tried actions with results
        - Dynamic optimization goal

        Args:
            action_space: Available actions
            context: Rich context for action selection
            top_k: Number of actions to select

        Returns:
            ActionSelectionResult with selected actions
        """
        if not self._llm_client:
            return self._fallback_select(action_space, top_k)

        if len(action_space) <= top_k:
            return ActionSelectionResult(selected_actions=list(action_space))

        prompt = self._build_rich_prompt(action_space, context, top_k)

        for _ in range(self._max_retries + 1):
            try:
                response = self._llm_client.generate(
                    system_prompt=self._get_rich_system_prompt(),
                    user_prompt=prompt,
                    temperature=self._temperature,
                )

                result = self._parse_response(action_space, top_k, response)
                if result.selected_actions:
                    return result

            except Exception:
                continue

        return self._fallback_select(action_space, top_k)

    def select(
        self,
        action_space: ActionSpace,
        context: Dict[str, Any],
        top_k: int = 10,
        optimize_goal: str = "quality",
    ) -> ActionSelectionResult:
        """
        Select top-k actions using LLM (legacy interface).

        Args:
            action_space: Available actions
            context: Context including current config, evaluation results, etc.
            top_k: Number of actions to select

        Returns:
            ActionSelectionResult with selected actions
        """
        if not self._llm_client:
            return self._fallback_select(action_space, top_k)

        if len(action_space) <= top_k:
            return ActionSelectionResult(selected_actions=list(action_space))

        prompt = self._build_prompt(action_space, context, top_k, optimize_goal)

        for _ in range(self._max_retries + 1):
            try:
                response = self._llm_client.generate(
                    system_prompt=self._get_system_prompt(),
                    user_prompt=prompt,
                    temperature=self._temperature,
                )

                result = self._parse_response(action_space, top_k, response)
                if result.selected_actions:
                    return result

            except Exception:
                continue

        return self._fallback_select(action_space, top_k)

    def rank(
        self,
        actions: List[Action],
        context: Dict[str, Any],
    ) -> List[Tuple[Action, float]]:
        """
        Rank actions by predicted effectiveness.

        Args:
            actions: Actions to rank
            context: Context for ranking

        Returns:
            List of (action, score) tuples, sorted by score descending
        """
        if not self._llm_client or not actions:
            return [(a, 0.5) for a in actions]

        prompt = self._build_ranking_prompt(actions, context)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a pipeline optimization expert.",
                user_prompt=prompt,
                temperature=0.1,
            )

            scores = self._parse_ranking_response(response, actions)
            ranked = [(a, scores.get(self._action_key(a), 0.5)) for a in actions]
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked

        except Exception:
            return [(a, 0.5) for a in actions]

    def _get_system_prompt(self) -> str:
        """Get system prompt for action selection."""
        return """You are an expert at optimizing data processing pipelines.
Your task is to select the most promising optimization actions based on the current pipeline state.

Consider:
1. Current pipeline quality and cost metrics
2. Type of operators in the pipeline
3. Applicability of each directive to each operator
4. Optimization goal (quality improvement or cost reduction)

Output your selection as a JSON object with the following format:
{
  "selected": [
    {
      "directive_template": "tighten_threshold",
      "target_locator": {"operator_id": "op-123", "audit_identity_hash": "hash"},
      "instantiate_params": {"intensity": 0.2},
      "directive": "tighten_filters",
      "reason": "brief reason",
      "score": 0.0
    },
    ...
  ],
  "reasoning": "Overall strategy explanation"
}"""

    def _get_rich_system_prompt(self) -> str:
        """Get rich system prompt for action selection with full context."""
        return """You are an expert at optimizing data processing pipelines.
Your task is to select the most promising optimization actions based on the current pipeline state and execution history.

You will be provided with:
1. Current pipeline configuration and metrics
2. Execution results sample (showing actual outputs)
3. Data sample (input data characteristics)
4. Action performance history (which actions have worked well)
5. Already tried actions in this optimization path
6. Current optimization goal

Your goal is to balance:
- Exploitation: Select actions that have historically performed well
- Exploration: Try actions that haven't been tested yet
- Goal alignment: Focus on improving the current optimization target

Output your selection as a JSON object:
{
  "selected": [
    {
      "directive_template": "tighten_threshold",
      "target_locator": {"operator_id": "op-123", "audit_identity_hash": "hash"},
      "instantiate_params": {"intensity": 0.2},
      "directive": "tighten_filters",
      "reason": "brief reason",
      "score": 0.0
    },
    ...
  ],
  "reasoning": "Overall strategy explanation"
}"""

    def _build_rich_prompt(
        self,
        action_space: ActionSpace,
        context: ActionSelectionContext,
        top_k: int,
    ) -> str:
        """Build rich prompt with full context for action selection."""
        import yaml

        actions_desc = []
        for i, action in enumerate(action_space):
            actions_desc.append(
                f"  [{i}] Operator: {action.operator_name} "
                f"(id {action.target_locator.operator_id}), "
                f"Template: {action.directive_template}, "
                f"Directive: {action.directive_name}, "
                f"Target: {action.target_locator.to_dict()}, "
                f"Params: {json.dumps(action.instantiate_params, ensure_ascii=False)}"
            )

        config_yaml = yaml.dump(context.config, allow_unicode=True, default_flow_style=False)
        if len(config_yaml) > 3000:
            config_yaml = config_yaml[:3000] + "\n... (truncated)"

        prompt = f"""Analyze the pipeline and select the {top_k} most promising optimization actions.

**Current Pipeline Configuration:**
```yaml
{config_yaml}
```

**Current Metrics:**
- Quality: {context.quality:.4f}
- Cost: ${context.cost:.2f}

**Execution Results Sample:**
{context.get_execution_sample_summary(max_chars=2000)}

**Data Sample:**
{context.get_data_sample_summary(max_chars=1000)}

**Action Performance History:**
{context.action_stats.get_summary()}

**Already Tried in This Path:**
{context.get_tried_action_summary(limit=10)}

**Optimization Goal:** {context.optimize_goal}

**Iteration:** {context.iteration + 1} / {context.max_iterations}

**Available Actions ({len(action_space)} total):**
{chr(10).join(actions_desc)}

Select {top_k} actions that are most likely to improve {context.optimize_goal}.

Consider:
1. Which operators have the most room for improvement?
2. Which directives have worked well historically?
3. What hasn't been tried yet in this path?
4. Is the current optimization goal quality or cost?

Prioritize exploration of untested actions while balancing with exploitation of proven performers."""

        return prompt

    def _build_prompt(
        self,
        action_space: ActionSpace,
        context: Dict[str, Any],
        top_k: int,
        optimize_goal: str,
    ) -> str:
        """Build the action selection prompt."""
        actions_desc = []
        for i, action in enumerate(action_space):
            actions_desc.append(
                f"  [{i}] Operator: {action.operator_name} "
                f"(id {action.target_locator.operator_id}), "
                f"Template: {action.directive_template}, "
                f"Directive: {action.directive_name}, "
                f"Target: {action.target_locator.to_dict()}, "
                f"Params: {json.dumps(action.instantiate_params, ensure_ascii=False)}"
            )

        current_config = context.get("current_config", {})
        current_quality = context.get("quality", "unknown")
        current_cost = context.get("cost", "unknown")
        previous_attempts = context.get("previous_attempts", [])

        prompt = f"""Analyze the pipeline and select the {top_k} most promising optimization actions.

**Current Pipeline State:**
- Process steps: {len(current_config.get("process", []))} operators
- Quality score: {current_quality}
- Cost: {current_cost}

**Available Actions ({len(action_space)} total):**
{chr(10).join(actions_desc)}

**Previous Attempts:**
{json.dumps(previous_attempts[-5:], indent=2) if previous_attempts else "None"}

**Optimization Goal:** {optimize_goal}

Select {top_k} actions that are most likely to improve the pipeline.
Focus on actions that address specific weaknesses or opportunities."""

        return prompt

    def _build_ranking_prompt(
        self,
        actions: List[Action],
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for ranking actions."""
        actions_desc = [
            f"[{i}] {a.operator_name}[{a.target_locator.operator_id}] "
            f"-> {a.directive_template} ({a.directive_name})"
            for i, a in enumerate(actions)
        ]

        return f"""Rank these optimization actions by predicted effectiveness.

Actions:
{chr(10).join(actions_desc)}

Context: Quality={context.get("quality", "unknown")}, Cost={context.get("cost", "unknown")}

Output a JSON object mapping action index to predicted effectiveness score (0.0-1.0):
{{"0": 0.8, "1": 0.5, ...}}"""

    def _parse_response(
        self,
        action_space: ActionSpace,
        top_k: int,
        response: str,
    ) -> ActionSelectionResult:
        """Parse LLM response into selected actions."""
        data = self._parse_json_object(response)

        selected = []
        scores = {}
        planned: List[PlannedActionSelection] = []
        seen_action_keys = set()

        action_map = {(a.target_locator.operator_id, a.directive_name): a for a in action_space}
        action_map_by_key = {a.action_key: a for a in action_space}
        action_map_by_template = {
            (
                a.target_locator.operator_id,
                a.target_locator.audit_identity_hash,
                a.directive_template,
            ): a
            for a in action_space
        }
        action_map_by_template_op = {
            (a.target_locator.operator_id, a.directive_template): a for a in action_space
        }
        ordered_actions = list(action_space)

        for item in data.get("selected", [])[:top_k]:
            action = self._resolve_selected_action(
                item,
                ordered_actions,
                action_map,
                action_map_by_key,
                action_map_by_template,
                action_map_by_template_op,
            )

            if action is None or action.action_key in seen_action_keys:
                continue

            selected.append(action)
            seen_action_keys.add(action.action_key)

            score = float(item.get("score", 1.0))
            scores[action.action_key] = score

            planned_item = self._to_planned_selection(item, fallback_action=action, score=score)
            if planned_item is not None:
                planned.append(planned_item)

        if not planned:
            planned = self._plans_from_actions(selected)

        return ActionSelectionResult(
            selected_actions=selected,
            scores=scores,
            reasoning=data.get("reasoning", ""),
            planned_selections=planned,
        )

    def _resolve_selected_action(
        self,
        item: Dict[str, Any],
        ordered_actions: List[Action],
        action_map: Dict[Tuple[str, str], Action],
        action_map_by_key: Dict[str, Action],
        action_map_by_template: Dict[Tuple[str, str, str], Action],
        action_map_by_template_op: Dict[Tuple[str, str], Action],
    ) -> Optional[Action]:
        """Resolve a selected JSON item to a concrete action."""
        action_key = item.get("action_key")
        if isinstance(action_key, str) and action_key in action_map_by_key:
            return action_map_by_key[action_key]

        op_id: Optional[str] = item.get("operator_id")
        locator = self._parse_target_locator(item.get("target_locator"))
        if locator is not None:
            op_id = locator.operator_id

        if op_id is None and isinstance(item.get("operator_index"), int):
            index_value = item.get("operator_index")
            if 0 <= index_value < len(ordered_actions):
                op_id = ordered_actions[index_value].target_locator.operator_id
                locator = ordered_actions[index_value].target_locator

        directive_name = item.get("directive") or item.get("directive_name")
        directive_template = item.get("directive_template") or self._map_directive_to_template(
            directive_name
        )

        if op_id is None:
            return None

        if locator is not None and directive_template:
            template_key = (op_id, locator.audit_identity_hash, directive_template)
            if template_key in action_map_by_template:
                return action_map_by_template[template_key]

        if directive_name:
            legacy_key = (op_id, directive_name)
            if legacy_key in action_map:
                return action_map[legacy_key]

        if directive_template:
            template_op_key = (op_id, directive_template)
            if template_op_key in action_map_by_template_op:
                return action_map_by_template_op[template_op_key]

        return None

    def _parse_target_locator(self, payload: Any) -> Optional[TargetLocator]:
        """Parse target locator payload into object form."""
        if not isinstance(payload, dict):
            return None

        operator_id = payload.get("operator_id")
        audit_identity_hash = payload.get("audit_identity_hash")
        if not isinstance(operator_id, str) or not operator_id:
            return None
        if not isinstance(audit_identity_hash, str) or not audit_identity_hash:
            return None

        return TargetLocator(
            operator_id=operator_id,
            audit_identity_hash=audit_identity_hash,
        )

    def _to_planned_selection(
        self,
        item: Dict[str, Any],
        fallback_action: Action,
        score: float,
    ) -> Optional[PlannedActionSelection]:
        """Build a structured plan item from parsed model output."""
        target_locator = self._parse_target_locator(item.get("target_locator")) or fallback_action.target_locator
        if target_locator is None:
            return None

        directive_name = item.get("directive") or item.get("directive_name") or fallback_action.directive_name
        directive_template = (
            item.get("directive_template")
            or self._map_directive_to_template(directive_name)
            or fallback_action.directive_template
        )
        instantiate_params = item.get("instantiate_params") or item.get("params")
        if not isinstance(instantiate_params, dict):
            instantiate_params = dict(fallback_action.instantiate_params)

        return PlannedActionSelection(
            directive_template=directive_template,
            target_locator=target_locator,
            instantiate_params=dict(instantiate_params),
            directive_name=directive_name,
            score=score,
            reason=str(item.get("reason", "")),
        )

    def _map_directive_to_template(self, directive_name: Optional[str]) -> str:
        """Map legacy directive names to canonical directive template names."""
        if not directive_name:
            return ""

        alias_map = {
            "tighten_filters": "tighten_threshold",
            "loosen_filters": "loosen_threshold",
            "remove_redundant_ops": "remove_redundant_op",
            "reorder_filters_first": "safe_reorder_local",
        }
        return alias_map.get(directive_name, directive_name)

    def _plans_from_actions(self, actions: List[Action]) -> List[PlannedActionSelection]:
        """Build fallback structured plan payload from selected actions."""
        return [
            PlannedActionSelection(
                directive_template=action.directive_template,
                target_locator=action.target_locator,
                instantiate_params=dict(action.instantiate_params),
                directive_name=action.directive_name,
                score=1.0,
            )
            for action in actions
        ]

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        """Parse JSON object from plain or fenced model output."""
        text = response.strip()
        candidates: List[str] = [text]

        if "```json" in text:
            candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
        if "```" in text:
            candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(text[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data

        raise ValueError("Unable to parse JSON object from LLM response")

    def _parse_ranking_response(
        self,
        response: str,
        actions: List[Action],
    ) -> Dict[str, float]:
        """Parse ranking response."""
        data = self._parse_json_object(response)

        scores = {}
        for i, action in enumerate(actions):
            key = str(i)
            scores[self._action_key(action)] = float(data.get(key, 0.5))

        return scores

    def _action_key(self, action: Action) -> str:
        """Generate a unique key for an action."""
        return action.action_key

    def _fallback_select(
        self,
        action_space: ActionSpace,
        top_k: int,
    ) -> ActionSelectionResult:
        """Fallback selection when LLM is not available."""
        return ActionSelectionResult(
            selected_actions=list(action_space)[:top_k],
            reasoning="LLM not available, using first-k fallback",
            planned_selections=self._plans_from_actions(list(action_space)[:top_k]),
        )


__all__ = [
    "LLMActionSelector",
    "ActionSelectionResult",
    "PlannedActionSelection",
]
