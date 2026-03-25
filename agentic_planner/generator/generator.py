# -*- coding: utf-8 -*-
"""Two-step NL generator: select operators → fill params → DJ executable dict."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from agentic_planner.contracts.plan_bridge import OperatorStep, plan_operators_to_process
from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.generator.catalog import build_operator_catalog_text, build_operator_detail_text
from agentic_planner.generator.llm import LLMJsonClient, parse_json_object_strict
from agentic_planner.generator.op_schema import (
    build_schema_block,
    format_allowlist_for_prompt,
    get_init_param_allowlist,
    sanitize_params,
    validate_params_bind,
)
from agentic_planner.generator.prompts import (
    FILL_BATCH_SYSTEM,
    FILL_BATCH_USER_TEMPLATE,
    FILL_ONE_SYSTEM,
    FILL_ONE_USER_TEMPLATE,
    SELECT_SYSTEM,
    SELECT_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from agentic_planner.generator.embedding.backend import EmbeddingBackend

logger = logging.getLogger(__name__)

RetrievalMode = Literal["none", "bm25", "vector"]


def _get_operators_registry() -> Dict[str, Any]:
    """Get the operators registry from data_juicer."""
    try:
        from data_juicer.ops.base_op import OPERATORS
        return OPERATORS.modules
    except ImportError:
        return {}


def assemble_executable_config(
    *,
    dataset_path: str,
    export_path: str,
    operators: List[OperatorStep],
    extra_config: Optional[Dict[str, Any]] = None,
) -> DJExecutableConfig:
    """Build a DJ executable dict from operator steps + IO paths."""
    cfg: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "export_path": export_path,
        "process": plan_operators_to_process(operators),
    }
    if extra_config:
        for k, v in extra_config.items():
            if k in ("dataset_path", "export_path", "process"):
                continue
            cfg[k] = v
    return cfg


FillMode = Literal["per_operator", "batch"]


class NLRecipeGenerator:
    """
    Generate a DJ executable ``dict`` from natural language + dataset hints.

    Steps:
        1. ``operator_names`` via LLM (optionally narrowed by retrieval).
        2. Fill parameters using **strict allowlists** from each operator's ``__init__``
           signature (no hallucinated keys). Default: one LLM call per operator
           (``fill_mode="per_operator"``) for smaller models.
        3. Sanitize (drop unknown keys), optional bind check, merge into recipe dict.

    Retrieval modes:
        - ``none``: Use full operator catalog (default, for large LLMs).
        - ``bm25``: Use BM25 keyword matching to narrow candidates.
        - ``vector``: Use vector embedding similarity to narrow candidates.
    """

    def __init__(
        self,
        llm: LLMJsonClient,
        *,
        retrieval_mode: RetrievalMode = "none",
        embedder: Optional[EmbeddingBackend] = None,
        cache_dir: str = ".embedding_cache",
    ) -> None:
        """
        Initialize generator.

        :param llm: LLM client for JSON completion.
        :param retrieval_mode: "none" | "bm25" | "vector".
        :param embedder: Embedding backend (required for "vector" mode).
        :param cache_dir: Cache directory for vector index.
        """
        self._llm = llm
        self._retrieval_mode = retrieval_mode
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._vector_retriever = None

        if retrieval_mode == "vector" and embedder is None:
            raise ValueError("embedder is required when retrieval_mode='vector'")

    def generate(
        self,
        *,
        user_intent: str,
        dataset_path: str,
        export_path: str,
        dataset_hint: str = "",
        extra_config: Optional[Dict[str, Any]] = None,
        fill_mode: FillMode = "per_operator",
        strict_params: bool = True,
        candidate_top_k: int = 20,
    ) -> DJExecutableConfig:
        """
        Build executable config.

        :param fill_mode: ``per_operator`` (default, strongest constraint) or ``batch``
            (single JSON for all ops; still sanitized).
        :param strict_params: If True (default), drop any param key not in the operator
            signature and validate binding to ``__init__``.
        :param candidate_top_k: Max operators in narrowed catalog (when retrieval enabled).
        """
        catalog = self._build_catalog(
            user_intent=user_intent,
            dataset_hint=dataset_hint,
            top_k=candidate_top_k,
        )
        hint = dataset_hint or dataset_path
        sel = self._select_operators(user_intent, hint, catalog)
        if fill_mode == "per_operator":
            ops = self._fill_operators_per_operator(user_intent, hint, sel, strict_params)
        else:
            ops = self._fill_operators_batch(user_intent, hint, sel, strict_params)
        cfg = assemble_executable_config(
            dataset_path=dataset_path,
            export_path=export_path,
            operators=ops,
            extra_config=extra_config,
        )
        errors = validate_executable_config(cfg)
        if errors:
            raise ValueError("Generated config failed validation: " + "; ".join(errors))
        return cfg

    def _build_catalog(
        self,
        user_intent: str,
        dataset_hint: str,
        top_k: int,
    ) -> str:
        """Build operator catalog, optionally narrowed by retrieval."""
        if self._retrieval_mode == "none":
            return build_operator_catalog_text()

        if self._retrieval_mode == "bm25":
            return self._build_catalog_bm25(user_intent, dataset_hint, top_k)

        if self._retrieval_mode == "vector":
            return self._build_catalog_vector(user_intent, dataset_hint, top_k)

        return build_operator_catalog_text()

    def _build_catalog_bm25(
        self,
        user_intent: str,
        dataset_hint: str,
        top_k: int,
    ) -> str:
        """Build catalog narrowed by BM25 retrieval."""
        from agentic_planner.generator.candidate_filter import detect_modalities, filter_ops_by_modality
        from agentic_planner.generator.candidate_ranker import rank_candidates

        try:
            from data_juicer.tools.op_search import OPSearcher
        except ImportError:
            return build_operator_catalog_text()

        searcher = OPSearcher(include_formatter=False)
        all_ops = searcher.op_records
        modalities = detect_modalities(user_intent, dataset_hint)
        filtered = filter_ops_by_modality(all_ops, modalities)
        if not filtered:
            filtered = list(all_ops)

        top_names = rank_candidates(
            user_intent,
            filtered,
            top_k=max(1, top_k),
            dataset_hint=dataset_hint,
        )
        if not top_names:
            top_names = [rec.name for rec in filtered[: max(1, top_k)]]

        logger.info(f"BM25 narrowed to {len(top_names)} operators: {top_names[:5]}...")
        return build_operator_catalog_text(only_names=set(top_names))

    def _build_catalog_vector(
        self,
        user_intent: str,
        dataset_hint: str,
        top_k: int,
    ) -> str:
        """Build catalog narrowed by vector retrieval."""
        from agentic_planner.generator.candidate_retriever import VectorRetriever

        if self._vector_retriever is None:
            self._vector_retriever = VectorRetriever(
                embedder=self._embedder,
                cache_dir=self._cache_dir,
            )

        top_names = self._vector_retriever.retrieve(
            intent=user_intent,
            top_k=top_k,
            dataset_hint=dataset_hint,
        )

        if not top_names:
            logger.warning("Vector retrieval returned empty, falling back to full catalog")
            return build_operator_catalog_text()

        logger.info(f"Vector retrieval narrowed to {len(top_names)} operators: {top_names[:5]}...")
        return build_operator_catalog_text(only_names=set(top_names))

    def _select_operators(self, intent: str, dataset_hint: str, catalog: str) -> List[str]:
        operators = _get_operators_registry()
        user = SELECT_USER_TEMPLATE.format(
            intent=intent.strip(),
            dataset_hint=dataset_hint.strip(),
            catalog=catalog,
        )
        raw = self._llm.complete_json(SELECT_SYSTEM, user)
        names = raw.get("operator_names")
        if not isinstance(names, list) or not names:
            raise ValueError("LLM must return non-empty operator_names list")
        out: List[str] = []
        for n in names:
            name = str(n).strip()
            if name and name in operators:
                out.append(name)
        if not out:
            raise ValueError("No valid operator names after registry check")
        return out

    def _fill_operators_per_operator(
        self,
        intent: str,
        dataset_hint: str,
        operator_names: List[str],
        strict_params: bool,
    ) -> List[OperatorStep]:
        steps: List[OperatorStep] = []
        for op_name in operator_names:
            allow = get_init_param_allowlist(op_name)
            allow_csv = ", ".join(allow) if allow else ""
            sig_block = format_allowlist_for_prompt(op_name)
            detail = build_operator_detail_text([op_name])
            detail_snippet = detail[:8000] if detail else "(no retrieved docs)"

            user = FILL_ONE_USER_TEMPLATE.format(
                intent=intent.strip(),
                dataset_hint=dataset_hint.strip(),
                op_name=op_name,
                allowlist_csv=allow_csv,
                signature_block=sig_block,
                detail_snippet=detail_snippet,
            )
            raw = self._llm.complete_json(FILL_ONE_SYSTEM, user)
            params = raw.get("params", raw)
            if not isinstance(params, dict):
                params = {}
            if strict_params:
                params = sanitize_params(op_name, params)
                ok, msg = validate_params_bind(op_name, params)
                if not ok:
                    raise ValueError(
                        f"params for {op_name!r} failed signature bind after sanitize: {msg}",
                    )
            steps.append(OperatorStep(name=op_name, params=params))
        return steps

    def _fill_operators_batch(
        self,
        intent: str,
        dataset_hint: str,
        operator_names: List[str],
        strict_params: bool,
    ) -> List[OperatorStep]:
        operators = _get_operators_registry()
        schema_block = build_schema_block(operator_names)
        user = FILL_BATCH_USER_TEMPLATE.format(
            intent=intent.strip(),
            dataset_hint=dataset_hint.strip(),
            schema_block=schema_block,
        )
        raw = self._llm.complete_json(FILL_BATCH_SYSTEM, user)
        rows = raw.get("operators")
        if not isinstance(rows, list) or not rows:
            raise ValueError("LLM must return non-empty operators list")
        steps: List[OperatorStep] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            params = item.get("params", {})
            if not isinstance(params, dict):
                params = {}
            if not name or name not in operators:
                continue
            if strict_params:
                params = sanitize_params(name, params)
                ok, msg = validate_params_bind(name, params)
                if not ok:
                    raise ValueError(
                        f"params for {name!r} failed signature bind after sanitize: {msg}",
                    )
            steps.append(OperatorStep(name=name, params=params))
        if not steps:
            raise ValueError("No valid operators after parsing LLM JSON")
        return steps


def generate_recipe_from_llm_json_text(
    *,
    text: str,
    dataset_path: str,
    export_path: str,
    extra_config: Optional[Dict[str, Any]] = None,
    strict_params: bool = True,
) -> DJExecutableConfig:
    """Helper: parse a single JSON object from ``text`` and assemble; optionally sanitize."""
    operators = _get_operators_registry()
    data = parse_json_object_strict(text)
    rows = data.get("operators")
    if not isinstance(rows, list):
        raise ValueError("operators must be a list")
    steps: List[OperatorStep] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        params = item.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if name and name in operators:
            if strict_params:
                params = sanitize_params(name, params)
            steps.append(OperatorStep(name=name, params=params))
    if not steps:
        raise ValueError("no valid operators")
    return assemble_executable_config(
        dataset_path=dataset_path,
        export_path=export_path,
        operators=steps,
        extra_config=extra_config,
    )