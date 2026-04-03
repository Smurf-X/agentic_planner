# Agentic Planner

Natural language pipeline design and optimization for [Data-Juicer](https://github.com/modelscope/data-juicer).

## Overview

Agentic Planner is an independent Python package that provides:

1. **Generator**: Convert natural language descriptions to Data-Juicer YAML configurations
2. **Optimizer**: Optimize pipeline configurations using directives and search-based methods

## Installation

```bash
# Basic installation
pip install agentic-planner

# With vector retrieval support
pip install agentic-planner[vector]

# With Chinese tokenization support
pip install agentic-planner[jieba]

# Full installation
pip install agentic-planner[all]
```

### Requirements

- Python 3.10+
- [Data-Juicer](https://github.com/modelscope/data-juicer) (installed separately)

```bash
pip install data-juicer
```

## Quick Start

### Agent Runtime TUI (MVP)

```bash
python -m apps.tui.main
```

The TUI delegates runtime actions through `agent_runtime.api.service.AgentRuntimeService`.
Each runtime action emits one JSONL telemetry event to `.agent_runtime/logs/events.jsonl`.

### Generator: Natural Language to YAML

```python
from agentic_planner.generator import (
    NLRecipeGenerator,
    OpenAICompatibleJsonClient,
)

# Create LLM client
llm = OpenAICompatibleJsonClient(
    model="gpt-4o-mini",
    api_key="your-api-key",
)

# Create generator
generator = NLRecipeGenerator(llm=llm)

# Generate pipeline from natural language
config = generator.generate(
    user_intent="Filter short texts and remove duplicates",
    dataset_path="input.jsonl",
    export_path="output.jsonl",
)

# Save to YAML
import yaml
with open("pipeline.yaml", "w") as f:
    yaml.dump(config, f)
```

### Optimizer: Directive-based Optimization

```python
from agentic_planner.contracts import load_executable_config
from agentic_planner.optimizer import get_directive_engine

# Load config
config = load_executable_config("pipeline.yaml")

# Apply static directives
DirectiveEngine = get_directive_engine()
engine = DirectiveEngine.from_dict({
    "mode": "static",
    "directives": [
        "reorder_filters_first",
        "remove_redundant_ops",
    ],
})

result = engine.run(config)
if result.ok:
    print("Optimized!")
    # Save optimized config
    from agentic_planner.contracts import save_executable_config
    save_executable_config(result.config, "optimized.yaml")
```

### Optimizer: Search-based Optimization

```python
from agentic_planner.optimizer.search import BeamSearchStrategy

# Create search strategy with evaluator
search = BeamSearchStrategy(
    beam_width=4,
    max_iterations=5,
    evaluator=my_evaluator,  # Your quality evaluator
)

# Run optimization
result = search.optimize(config)
print(f"Best quality: {result.best_quality}")
```

## Features

### Generator

- **Natural language to YAML**: Convert user intent to executable pipeline configurations
- **Candidate retrieval**: BM25 or vector-based operator selection
- **Strict parameter validation**: No hallucinated parameters
- **Multi-language support**: Chinese and English

### Optimizer

#### Directive-based (Stage 1)
- `reorder_filters_first`: Move filters before mappers (predicate pushdown)
- `remove_redundant_ops`: Remove duplicate or no-op operators
- `adjust_threshold`: Tune filter thresholds
- `swap_model`: Change LLM model for operators
- `add_gleaning`: Add gleaning iterations for map operators
- `rewrite_prompt`: Optimize LLM prompts

#### Search-based (Stage 2)
- Greedy search
- Random search
- Beam search with Pareto front

### Stable Operator Identification

Use `OpLocator` for position-independent operator targeting:

```python
from agentic_planner.optimizer import OpLocator, ProcessIndex

# Build index from pipeline
index = ProcessIndex.build(config["process"])

# Find by type and parameter
locator = OpLocator(
    op_type="text_length_filter",
    param_match={"min_len": 10},
)
idx = index.locate(locator)
```

## Architecture

```
agentic_planner/
├── contracts/          # Core data structures
│   ├── recipe.py       # DJExecutableConfig
│   ├── cost.py         # CostBreakdown
│   ├── eval_protocol.py
│   └── plan_bridge.py  # OperatorStep
├── generator/          # NL to YAML generation
│   ├── generator.py    # NLRecipeGenerator
│   ├── catalog.py      # Operator catalog
│   ├── op_schema.py    # Parameter validation
│   ├── prompts.py      # LLM prompts
│   ├── embedding/      # Vector embeddings
│   └── ...
├── optimizer/          # Pipeline optimization
│   ├── op_locator.py   # Stable operator ID
│   ├── directives/     # Optimization directives
│   ├── search/         # Search strategies
│   └── ...
└── __init__.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black agentic_planner/
isort agentic_planner/

# Type check
mypy agentic_planner/
```

## License

Apache License 2.0
