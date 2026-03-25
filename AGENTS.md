# AGENTS.md

Coding agent instructions for the Agentic Planner repository.

## Project Overview

Agentic Planner is a Python package for natural language pipeline design and optimization for Data-Juicer. It provides:
- **Generator**: Convert natural language to Data-Juicer YAML configurations
- **Optimizer**: Optimize pipeline configurations using directives and search methods

## Build / Lint / Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_generator.py

# Run a single test function
pytest tests/test_generator.py::test_select_operators

# Run tests with verbose output
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=agentic_planner

# Format code (line-length 100)
black agentic_planner/
isort agentic_planner/

# Type check
mypy agentic_planner/

# Lint with ruff
ruff check agentic_planner/

# Run all quality checks
black agentic_planner/ && isort agentic_planner/ && ruff check agentic_planner/ && mypy agentic_planner/
```

## Code Style Guidelines

### File Headers

All Python files must start with:
```python
# -*- coding: utf-8 -*-
"""Module docstring."""
```

### Imports

Order imports in three groups, separated by blank lines:

```python
# -*- coding: utf-8 -*-
"""Module docstring."""

from __future__ import annotations  # First, if needed

import os  # Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # Third-party
import numpy as np
from pydantic import BaseModel

from agentic_planner.contracts.recipe import DJExecutableConfig  # Local imports
from agentic_planner.generator import NLRecipeGenerator
```

- Use `from __future__ import annotations` for modern type hints
- Use `TYPE_CHECKING` block for imports only needed for type hints:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_planner.optimizer.op_locator import ProcessIndex
```

### Formatting

- **Line length**: 100 characters (configured in pyproject.toml)
- **Formatter**: black with default settings
- **Import sorting**: isort with `profile = "black"`

### Type Hints

- Always use type hints for function parameters and return types
- Use `Optional[T]` for optional parameters, not `T | None` (Python 3.10 compatibility)
- Use `Dict[str, Any]` for flexible dictionary types
- Use `List[T]` instead of `list[T]`
- Use `Protocol` and `@runtime_checkable` for duck-typed interfaces:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: List[str]) -> np.ndarray: ...
```

- Use `Literal` for constrained string values:

```python
from typing import Literal
RetrievalMode = Literal["none", "bm25", "vector"]
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `generator.py`, `op_schema.py` |
| Classes | PascalCase | `NLRecipeGenerator`, `DirectiveResult` |
| Functions | snake_case | `generate_recipe()`, `validate_params_bind()` |
| Variables | snake_case | `operator_names`, `config_dict` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_MODEL`, `INDEX_VERSION` |
| Private methods | _leading_underscore | `_build_catalog()`, `_ensure_index()` |
| Type aliases | PascalCase | `DJExecutableConfig`, `RetrievalMode` |

### Data Classes and Models

- Use `@dataclass` for simple data containers:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DirectiveResult:
    ok: bool
    applied: bool
    directive_name: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
```

- Use Pydantic `BaseModel` for configuration with validation:

```python
from pydantic import BaseModel, Field

class SearchConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)
```

### Error Handling

- Return error lists for validation functions:

```python
def validate_executable_config(cfg: DJExecutableConfig) -> List[str]:
    """Returns a list of error messages (empty if ok)."""
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config must be a mapping"]
    # ... collect errors
    return errors
```

- Raise `ValueError` for invalid arguments:

```python
if not operator_names:
    raise ValueError("No valid operator names after registry check")
```

- Use result objects for operations that can fail:

```python
@dataclass
class DirectiveResult:
    ok: bool
    applied: bool
    message: str
    # ...
```

### Lazy Imports

Use lazy imports for optional dependencies to avoid hard dependency issues:

```python
def _get_operators_registry() -> Dict[str, Any]:
    """Get the operators registry from data_juicer."""
    try:
        from data_juicer.ops.base_op import OPERATORS
        return OPERATORS.modules
    except ImportError:
        return {}
```

### Docstrings

- Module-level docstring at the top of each file
- Function docstrings with parameter descriptions:

```python
def generate(
    self,
    *,
    user_intent: str,
    dataset_path: str,
    candidate_top_k: int = 20,
) -> DJExecutableConfig:
    """
    Build executable config.

    :param user_intent: User's natural language intent.
    :param dataset_path: Path to input dataset.
    :param candidate_top_k: Max operators in narrowed catalog.
    :return: Executable configuration dictionary.
    """
```

### Abstract Base Classes

Use ABC for abstract base classes:

```python
from abc import ABC, abstractmethod

class Directive(ABC):
    name: str = "base"

    @abstractmethod
    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: "ProcessIndex",
    ) -> DirectiveResult:
        pass
```

### Enums

Use `str, Enum` for string enums:

```python
from enum import Enum

class SearchStrategyType(str, Enum):
    GREEDY = "greedy"
    BEAM = "beam"
    RANDOM = "random"
```

## Project Structure

```
agentic_planner/
├── __init__.py          # Public API exports
├── contracts/           # Core data structures
│   ├── recipe.py        # DJExecutableConfig, load/save YAML
│   ├── cost.py          # CostBreakdown
│   ├── eval_protocol.py # Evaluation configuration
│   └── plan_bridge.py   # OperatorStep, process conversion
├── generator/           # Natural language to YAML
│   ├── generator.py     # NLRecipeGenerator (main class)
│   ├── catalog.py       # Operator catalog building
│   ├── op_schema.py     # Parameter validation
│   ├── prompts.py       # LLM prompt templates
│   ├── embedding/       # Vector embedding backends
│   ├── candidate_retriever.py  # Vector retrieval
│   └── candidate_ranker.py     # BM25 ranking
└── optimizer/           # Pipeline optimization
    ├── op_locator.py    # Stable operator identification
    ├── directive_engine.py
    ├── directives/      # Optimization directives
    └── search/          # Search strategies
```

## Key Patterns

### Two-Step Generation (Generator)

1. **Select operators**: LLM chooses from catalog (optionally narrowed by BM25/vector)
2. **Fill parameters**: LLM fills params with strict allowlist validation

### Parameter Validation (op_schema.py)

```python
# Get allowed parameter names from __init__ signature
allow = get_init_param_allowlist(op_name)

# Sanitize: drop unknown keys
params = sanitize_params(op_name, params)

# Validate: check binding
ok, msg = validate_params_bind(op_name, params)
```

### Configuration Types

- `DJExecutableConfig = Dict[str, Any]` - Executable pipeline config (YAML-serializable)
- `OperatorStep` - Named tuple with `name` and `params`
- `OptimizationConfig` - Pydantic model for optimizer settings

## Dependencies

- Python 3.10+
- Required: pydantic, pyyaml, httpx, rank-bm25, numpy
- Optional: sentence-transformers (vector), jieba (Chinese), data-juicer (operator registry)