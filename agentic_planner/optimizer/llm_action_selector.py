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
    {"operator_index": 0, "directive": "directive_name", "reason": "brief reason"},
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
    {"operator_index": 0, "directive": "directive_name", "reason": "brief reason"},
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
                f"  [{i}] Operator: {action.operator_name} (index {action.operator_index}), "
                f"Directive: {action.directive_name}"
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
                f"  [{i}] Operator: {action.operator_name} (index {action.operator_index}), "
                f"Directive: {action.directive_name}"
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
            f"[{i}] {a.operator_name}[{a.operator_index}] -> {a.directive_name}"
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
        text = response.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)

        selected = []
        scores = {}

        action_map = {(a.operator_index, a.directive_name): a for a in action_space}

        for item in data.get("selected", [])[:top_k]:
            op_idx = item.get("operator_index")
            directive = item.get("directive")

            key = (op_idx, directive)
            if key in action_map:
                selected.append(action_map[key])
                scores[f"{op_idx}:{directive}"] = 1.0

        return ActionSelectionResult(
            selected_actions=selected,
            scores=scores,
            reasoning=data.get("reasoning", ""),
        )

    def _parse_ranking_response(
        self,
        response: str,
        actions: List[Action],
    ) -> Dict[str, float]:
        """Parse ranking response."""
        text = response.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)

        scores = {}
        for i, action in enumerate(actions):
            key = str(i)
            scores[self._action_key(action)] = float(data.get(key, 0.5))

        return scores

    def _action_key(self, action: Action) -> str:
        """Generate a unique key for an action."""
        return f"{action.operator_index}:{action.directive_name}"

    def _fallback_select(
        self,
        action_space: ActionSpace,
        top_k: int,
    ) -> ActionSelectionResult:
        """Fallback selection when LLM is not available."""
        return ActionSelectionResult(
            selected_actions=list(action_space)[:top_k],
            reasoning="LLM not available, using first-k fallback",
        )


__all__ = [
    "LLMActionSelector",
    "ActionSelectionResult",
]
