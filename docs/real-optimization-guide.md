# 真实优化流程执行指南

本文档描述如何在 Linux 服务器上执行真实的 Data-Juicer Pipeline 优化流程。

## 1. 环境准备

### 1.1 安装 Data-Juicer

```bash
pip install py-data-juicer
```

### 1.2 安装 agentic-planner

```bash
cd /path/to/agentic_planner_new
pip install -e ".[all]"
```

### 1.3 验证安装

```bash
python -c "from data_juicer.ops.base_op import OPERATORS; print('DJ OK')"
python -c "from agentic_planner.optimizer.executor_adapter import DJExecutorAdapter; print('agentic_planner OK')"
```

## 2. 配置文件

### 2.1 模型配置 (`models.yaml`)

创建文件 `examples/models.yaml`：

```yaml
# Judge configuration (for LLM-as-Judge evaluation)
judge:
  model: "your-judge-model"
  api_key: "${YOUR_API_KEY}"
  base_url: "https://api.your-provider.com/v1"
  price_per_million: 0.15
  temperature: 0.1

# Pipeline models: models available for operators
pipeline_models:
  model-a:
    model: "model-a-name"
    api_key: "${YOUR_API_KEY}"
    base_url: "https://api.your-provider.com/v1"
    price_per_million: 0.1
    
  model-b:
    model: "model-b-name"
    api_key: "${YOUR_API_KEY}"
    base_url: "https://api.your-provider.com/v1"
    price_per_million: 0.5

# Candidate models: models to try during optimization
candidate_models:
  - model-a
  - model-b
```

### 2.2 设置环境变量

```bash
export YOUR_API_KEY="your-actual-api-key"
```

## 3. 测试数据

### 3.1 数据格式

数据文件为 JSONL 格式，每行一个 JSON 对象：

```json
{"text": "What is machine learning? Machine learning is a subset of artificial intelligence..."}
{"text": "How do neural networks work? Neural networks are computing systems..."}
{"text": "The theory of relativity, developed by Albert Einstein..."}
```

### 3.2 示例数据

数据文件位于 `data/qa_data.jsonl`，包含 50 条英文 QA 数据。

## 4. 执行命令

### 4.1 基本执行

```bash
python examples/run_optimizer.py \
  --models-config examples/models.yaml \
  --sample-size 5 \
  --beam-width 2 \
  --max-iterations 2
```

### 4.2 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models-config` | 模型配置文件路径 | 必填 |
| `--sample-size` | 评估采样数量 | 10 |
| `--beam-width` | Beam 宽度 | 3 |
| `--max-iterations` | 最大迭代次数 | 2 |
| `--llm-selection-top-k` | LLM 选择的动作数量 | 10 |
| `--no-llm-selection` | 禁用 LLM 动作选择 | False |
| `--stub` | 使用模拟执行器（测试用） | False |

### 4.3 完整执行示例

```bash
# 小规模测试
python examples/run_optimizer.py \
  --models-config examples/models.yaml \
  --sample-size 3 \
  --beam-width 1 \
  --max-iterations 1 \
  --llm-selection-top-k 3

# 中等规模
python examples/run_optimizer.py \
  --models-config examples/models.yaml \
  --sample-size 10 \
  --beam-width 3 \
  --max-iterations 3

# 大规模
python examples/run_optimizer.py \
  --models-config examples/models.yaml \
  --sample-size 50 \
  --beam-width 5 \
  --max-iterations 5
```

## 5. 预期输出

### 5.1 执行日志

```
============================================================
Pipeline Optimizer - End-to-End Example
============================================================

[1/7] Loading model configuration...
  - Config file: /path/to/models.yaml
  - Judge model: your-judge-model
  - Pipeline models: ['model-a', 'model-b']
  - Candidate models: ['model-a', 'model-b']

[2/7] Creating initial pipeline config...
  - Dataset: /path/to/data/qa_data.jsonl
  - Output: /path/to/output/optimized.jsonl
  - Operators: 2
  - Pipeline LLM model: gpt-4o-mini

[3/7] Loading data sample...
  - Sample size: 5

[4/7] Setting up evaluator...
  - Judge model: your-judge-model
  - Price table: 2 models
  - Executor: Data-Juicer (real execution)
  - Sample size: 5

[5/7] Configuring action space...
  - Directives: tighten_filters, loosen_filters
  - Model swap: 2 candidate models

[6/7] Setting up LLM Action Selector...
  - Enabled: True
  - Selector model: model-a
  - Top-k: 10

[7/7] Configuring BeamSearch strategy...
  - Beam width: 2
  - Max iterations: 2
  - LLM selection: True

============================================================
Running optimization...
============================================================

------------------------------------------------------------
[Results]
------------------------------------------------------------
  - Success: True
  - Total candidates: X
  - Total evaluations: X
  - Pareto front size: X

  Best by quality:
    - Quality: 0.XXXX
    - Cost: $0.XX
    - Origin: root+tighten_filters[text_length_filter]

  Best by cost:
    - Quality: 0.XXXX
    - Cost: $0.XX
    - Origin: root+loosen_filters[text_length_filter]

  Best balanced:
    - Quality: 0.XXXX
    - Cost: $0.XX
    - Origin: root

  Pareto front (X candidates):
    [1] Q=0.XXXX, C=$0.XX
    [2] Q=0.XXXX, C=$0.XX
    ...

[Saved files]
  - pareto_*.yaml in /path/to/output

============================================================
Optimization complete!
============================================================
```

### 5.2 输出文件

优化完成后，`output/` 目录下会生成：

- `pareto_1.yaml` - Pareto 前沿第 1 个配置
- `pareto_2.yaml` - Pareto 前沿第 2 个配置
- ...

每个 YAML 文件包含完整的 Pipeline 配置，可直接用 Data-Juicer 执行：

```bash
dj --config output/pareto_1.yaml
```

## 6. 优化流程说明

### 6.1 整体流程

```
1. 加载模型配置 → ModelRegistry
2. 创建初始 Pipeline 配置（2 个 filter 算子）
3. 加载数据样本
4. 设置评估器：
   - Judge LLM 评估质量
   - DJExecutorAdapter 真实执行 Pipeline
5. 构建动作空间：
   - tighten_filters: 收紧 filter 阈值
   - loosen_filters: 放宽 filter 阈值
   - swap_model: 换模型（对 LLM 算子）
6. 设置 LLM Action Selector
7. 运行 BeamSearch 优化
8. 输出 Pareto 前沿配置
```

### 6.2 动作选择流程

每次迭代时，LLM 会收到以下上下文：

- 当前 Pipeline 配置
- 当前质量和成本
- 执行结果样本
- 动作历史统计（哪些动作效果好）
- 已尝试的动作
- 当前优化目标（动态切换 quality/cost）

LLM 从候选动作中选择最有可能改进的动作。

### 6.3 评估流程

1. **执行 Pipeline**：DJExecutorAdapter 调用 Data-Juicer 执行配置
2. **收集输出**：获取处理后的数据
3. **LLM 评估**：Judge LLM 对输出质量打分
4. **计算成本**：根据 token 使用量和模型价格计算

## 7. 常见问题

### 7.1 DJ 执行器在 Windows 上有问题

DJ 的日志系统使用相对路径，在 Windows 上会生成包含 `..` 的文件名，导致创建失败。

**解决方案**：在 Linux/macOS 上运行。

### 7.2 如何添加新的优化指令

1. 在 `agentic_planner/optimizer/directives/` 创建新指令
2. 继承 `Directive` 基类
3. 实现 `apply_with_index()` 方法
4. 注册到 `DIRECTIVE_REGISTRY`

### 7.3 如何添加新的 LLM 模型

在 `models.yaml` 的 `pipeline_models` 中添加：

```yaml
pipeline_models:
  new-model:
    model: "new-model-name"
    api_key: "${NEW_MODEL_API_KEY}"
    base_url: "https://api.new-provider.com/v1"
    price_per_million: 1.0
```

## 8. 文件结构

```
agentic_planner_new/
├── data/
│   └── qa_data.jsonl          # 测试数据
├── examples/
│   ├── models.yaml            # 模型配置模板
│   └── run_optimizer.py       # 优化脚本
├── output/
│   └── pareto_*.yaml          # 输出配置
└── agentic_planner/
    └── optimizer/
        ├── action.py          # Action 和 ActionSpace
        ├── action_context.py  # LLM 选择上下文
        ├── executor_adapter.py # DJ 执行器适配
        ├── evaluator.py       # 评估器
        ├── llm_action_selector.py # LLM 动作选择器
        ├── model_registry.py  # 模型注册表
        └── search/
            └── beam.py        # BeamSearch 策略
```