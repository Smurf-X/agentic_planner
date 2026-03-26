# -*- coding: utf-8 -*-
"""Prompt templates for two-step NL recipe generation."""

SELECT_SYSTEM = """You are a Data-Juicer recipe planner. You ONLY output JSON.
Choose operator names from the provided catalog. Do not invent operators."""

SELECT_USER_TEMPLATE = """User intent:
{intent}

Dataset hint (path or schema summary):
{dataset_hint}

Operator catalog (one per line: name | type | tags | description):
{catalog}

Return JSON ONLY with this shape:
{{"operator_names": ["op_a", "op_b", ...]}}
The list must be non-empty and every name must appear exactly as in the catalog."""

# --- Strict fill: params MUST be a subset of allowlisted __init__ keys (see op_schema). ---

FILL_BATCH_SYSTEM = """You are a Data-Juicer recipe planner. You ONLY output JSON.
For EACH operator, the "params" object may contain ONLY keys listed under that operator's
"Allowed parameter names". Do NOT invent parameter names (e.g. do not use api_model unless it
appears in the allowlist). Do NOT copy parameter names from other frameworks.
If unsure, use an empty object {{}} for params."""

FILL_BATCH_USER_TEMPLATE = """User intent:
{intent}

Dataset hint:
{dataset_hint}

Operators in order (with STRICT parameter allowlists from Data-Juicer source code):
{schema_block}

Return JSON ONLY with this shape:
{{"operators": [{{"name": "op_name", "params": {{...}}}}, ...]}}
Rules:
- The "name" field must match each operator in order.
- Every key inside each "params" MUST appear in that operator's allowlist below.
- Omit optional parameters rather than guessing names."""

FILL_ONE_SYSTEM = """You are a Data-Juicer recipe planner. You ONLY output JSON.
The "params" object may contain ONLY keys listed under "Allowed parameter names".
Do not invent keys. If unsure, return an empty params object."""

FILL_ONE_USER_TEMPLATE = """User intent:
{intent}

Dataset hint:
{dataset_hint}

You are configuring ONE operator step.

Operator name: {op_name}

Allowed parameter names (ONLY these keys may appear in "params"):
{allowlist_csv}

Signature detail:
{signature_block}

Optional reference (may omit irrelevant parts):
{detail_snippet}

Return JSON ONLY:
{{"params": {{...}}}}
where every key in "params" is in the allowlist. Use {{}} if defaults are enough."""

# Legacy (weak) — kept for compatibility; prefer batch/one strict prompts above.
FILL_SYSTEM = FILL_BATCH_SYSTEM
FILL_USER_TEMPLATE = FILL_BATCH_USER_TEMPLATE