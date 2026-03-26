# Pipeline 评估优化方案

**日期**：2026-03-25
**状态**：已实现（固定采样）+ 待实施（其他优化）
**范围**：优化 Pipeline 评估过程中的成本和效率

---

## 背景

在 BeamSearch 搜索过程中，每次评估一个候选 Pipeline 都需要：

1. 采样输入数据
2. **端到端运行 Pipeline**（主要成本）
3. LLM-as-Judge 评估输出质量
4. 收集 Cost 指标

当 Pipeline 包含 LLM 算子时，端到端执行会产生实际的 API 调用成本。搜索过程中可能需要评估数十甚至上百个候选配置，成本累积显著。

---

## 已实现：固定采样机制

### 设计原理

**借鉴 docetl 的做法**：在优化开始前随机采样 N 条数据，之后的整个优化过程中都使用这固定的 N 条数据进行评估，而不是每次评估都重新随机采样。

### 使用方法

```python
from agentic_planner.contracts.eval_protocol import EvalConfig
from agentic_planner.optimizer.evaluator import RealPipelineEvaluator
from agentic_planner.optimizer.search.beam import BeamSearchConfig, BeamSearchStrategy

# 方式 1：使用默认采样（自动采样）
eval_config = EvalConfig(
    sample_size=50,       # 采样数量（默认 50）
    random_seed=42,       # 随机种子（可复现，默认 42）
    task_description="清洗英文文本数据",
)

# 方式 2：用户提供预采样数据
eval_config = EvalConfig(
    fixed_samples=my_sampled_data,  # 直接提供采样好的数据
    task_description="清洗英文文本数据",
)

# 创建评估器
evaluator = RealPipelineEvaluator(
    eval_config=eval_config,
    llm_client=llm_client,
    executor_adapter=executor_adapter,
)

# 创建搜索策略
strategy = BeamSearchStrategy(
    config=BeamSearchConfig(beam_width=4, max_iterations=3),
    evaluator=evaluator,
)

# 执行搜索（固定采样会在第一次评估前自动完成）
report = strategy.search(pipeline_config)
```

### 实现细节

1. **EvalConfig 新增字段**：
   ```python
   fixed_samples: Optional[List[Dict[str, Any]]] = None
   ```
   如果提供，则直接使用；否则在优化开始时自动采样。

2. **RealPipelineEvaluator 新增方法**：
   ```python
   def prepare_fixed_samples(self, dataset_path: Optional[str] = None) -> None:
       """在优化开始前准备固定样本"""
   ```

3. **搜索策略集成**：
   ```python
   # BeamSearchStrategy.search() 开头
   if self._evaluator is not None and hasattr(self._evaluator, "prepare_fixed_samples"):
       self._evaluator.prepare_fixed_samples(dataset_path)
   ```

### 优点

| 优点 | 说明 |
|------|------|
| **公平比较** | 所有候选 Pipeline 在相同数据上评估，结果可比 |
| **可复现** | 固定随机种子，优化过程可复现 |
| **减少随机性** | 消除采样带来的随机波动 |
| **简单高效** | 一次采样，全程复用 |

---

## 待实施优化方案

### 1. 缓存评估结果

**原理**：相同配置的评估结果可以复用，避免重复执行。

**实现方案**：

```python
from typing import Dict, Tuple
from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.contracts.recipe import DJExecutableConfig

class CachedEvaluator:
    """带缓存的评估器包装类。"""
    
    def __init__(self, evaluator, max_cache_size: int = 1000):
        self._evaluator = evaluator
        self._cache: Dict[str, Tuple[CostBreakdown, float]] = {}
        self._max_cache_size = max_cache_size
        self._hits = 0
        self._misses = 0
    
    def evaluate(self, cfg: DJExecutableConfig) -> Tuple[CostBreakdown, float]:
        """评估配置，优先使用缓存。"""
        cache_key = self._hash_config(cfg)
        
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        
        self._misses += 1
        result = self._evaluator.evaluate(cfg)
        
        # LRU 淘汰
        if len(self._cache) >= self._max_cache_size:
            self._evict_oldest()
        
        self._cache[cache_key] = result
        return result
    
    def _hash_config(self, cfg: DJExecutableConfig) -> str:
        """生成配置的稳定哈希。"""
        import hashlib
        import json
        content = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict_oldest(self):
        """淘汰最旧的缓存条目。"""
        # 简单实现：随机删除一个
        self._cache.pop(next(iter(self._cache)))
    
    @property
    def cache_stats(self) -> Dict[str, int]:
        """返回缓存统计信息。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            "cache_size": len(self._cache),
        }
```

**集成方式**：

```python
# 在 BeamSearchStrategy 中使用
from agentic_planner.optimizer.evaluator import CachedEvaluator

cached_evaluator = CachedEvaluator(
    evaluator=real_evaluator,
    max_cache_size=500,
)

strategy = BeamSearchStrategy(
    config=config,
    evaluator=cached_evaluator,  # 包装后的评估器
)
```

**优点**：
- 实现简单，对现有代码无侵入
- 对重复配置完全避免重复评估
- 可追踪缓存命中率

**注意**：
- 需要合理的缓存大小限制
- 配置哈希要稳定且唯一

---

### 2. 增量评估

**原理**：当 Pipeline 只修改了部分算子时，可以重用未变化部分的中间结果，只重新执行变化的部分。

**实现方案**：

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class IntermediateResult:
    """算子执行的中间结果。"""
    
    operator_index: int
    operator_name: str
    input_data: List[Dict[str, Any]]
    output_data: List[Dict[str, Any]]
    cost: CostBreakdown


@dataclass
class ExecutionTrace:
    """Pipeline 执行的完整追踪。"""
    
    config_hash: str
    intermediates: List[IntermediateResult] = field(default_factory=list)
    final_output: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: CostBreakdown = field(default_factory=CostBreakdown)


class IncrementalEvaluator:
    """支持增量评估的评估器。"""
    
    def __init__(self, evaluator, executor_adapter):
        self._evaluator = evaluator
        self._executor = executor_adapter
        self._trace_cache: Dict[str, ExecutionTrace] = {}
    
    def evaluate_with_change(
        self,
        old_config: DJExecutableConfig,
        new_config: DJExecutableConfig,
        changed_operator_indices: List[int],
    ) -> Tuple[CostBreakdown, float]:
        """
        增量评估：只重新执行变化的算子。
        
        Args:
            old_config: 修改前的配置
            new_config: 修改后的配置
            changed_operator_indices: 变化的算子索引列表
        
        Returns:
            (cost, quality) 元组
        """
        old_hash = self._hash_config(old_config)
        
        if old_hash not in self._trace_cache:
            # 没有缓存，执行完整评估
            return self._full_evaluate(new_config)
        
        old_trace = self._trace_cache[old_hash]
        
        # 找到最早变化的算子
        first_changed = min(changed_operator_indices) if changed_operator_indices else 0
        
        # 从该位置开始重新执行
        new_intermediates = old_trace.intermediates[:first_changed]
        
        if not new_intermediates:
            # 第一个算子就变了，需要从头执行
            input_data = self._load_sample_data()
        else:
            # 从上一个未变化的算子输出开始
            input_data = new_intermediates[-1].output_data
        
        # 依次执行变化后的算子
        process = new_config.get("process", [])
        current_cost = CostBreakdown()
        
        for i in range(first_changed, len(process)):
            step = process[i]
            op_name, params = self._parse_step(step)
            
            output_data, step_cost = self._executor.run_single_operator(
                op_name, params, input_data
            )
            
            new_intermediates.append(IntermediateResult(
                operator_index=i,
                operator_name=op_name,
                input_data=input_data,
                output_data=output_data,
                cost=step_cost,
            ))
            
            input_data = output_data
            current_cost = self._merge_cost(current_cost, step_cost)
        
        # 缓存新的执行追踪
        new_trace = ExecutionTrace(
            config_hash=self._hash_config(new_config),
            intermediates=new_intermediates,
            final_output=input_data,
            total_cost=current_cost,
        )
        self._trace_cache[new_trace.config_hash] = new_trace
        
        # 评估最终输出的质量
        quality = self._evaluate_quality(input_data)
        
        return current_cost, quality
    
    def _full_evaluate(self, config: DJExecutableConfig):
        """完整评估（无缓存时）。"""
        result = self._evaluator.evaluate(config)
        return result
```

**使用场景**：

```python
# 在 BeamSearch 中使用
class BeamSearchStrategy:
    def _evaluate_action(self, beam: BeamCandidate, action: Action):
        # 记录变化的算子
        changed_indices = [action.operator_index]
        
        # 增量评估
        cost, quality = self._incremental_evaluator.evaluate_with_change(
            old_config=beam.config,
            new_config=action.apply(beam.config).config_after,
            changed_operator_indices=changed_indices,
        )
        return cost, quality
```

**优点**：
- 显著减少重复计算
- 特别适合单算子修改的场景

**限制**：
- 实现复杂度较高
- 需要执行器支持单算子执行
- 需要存储中间结果（内存开销）

---

### 3. 早停 + 剪枝

**原理**：在评估过程中提前判断候选配置的质量，如果明显差于当前最优，提前终止评估以节省成本。

**实现方案**：

```python
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class EarlyStopConfig:
    """早停配置。"""
    
    min_sample_size: int = 10
    """最小评估样本数（用于初步判断）。"""
    
    early_stop_threshold: float = 0.7
    """早停阈值：如果初步质量 < 最优质量 * threshold，则提前终止。"""
    
    enable_cost_early_stop: bool = True
    """是否启用成本早停。"""
    
    max_cost_ratio: float = 2.0
    """最大成本比例：如果当前成本 > 最优成本 * ratio，则提前终止。"""


class EarlyStopEvaluator:
    """支持早停的评估器。"""
    
    def __init__(
        self,
        evaluator,
        config: EarlyStopConfig,
        get_current_best: Callable[[], Optional[Tuple[CostBreakdown, float]]],
    ):
        self._evaluator = evaluator
        self._config = config
        self._get_current_best = get_current_best
        self._early_stops = 0
    
    def evaluate(
        self,
        cfg: DJExecutableConfig,
    ) -> Tuple[CostBreakdown, float]:
        """评估配置，支持早停。"""
        current_best = self._get_current_best()
        
        if current_best is None:
            # 没有最优结果，执行完整评估
            return self._full_evaluate(cfg)
        
        best_cost, best_quality = current_best
        
        # 第一阶段：小样本快速评估
        partial_cost, partial_quality = self._partial_evaluate(
            cfg, sample_size=self._config.min_sample_size
        )
        
        # 质量早停检查
        if partial_quality < best_quality * self._config.early_stop_threshold:
            self._early_stops += 1
            return partial_cost, partial_quality
        
        # 成本早停检查
        if self._config.enable_cost_early_stop:
            if partial_cost.llm_token_cost > best_cost.llm_token_cost * self._config.max_cost_ratio:
                self._early_stops += 1
                return partial_cost, partial_quality
        
        # 继续完整评估
        return self._full_evaluate(cfg)
    
    def _partial_evaluate(
        self,
        cfg: DJExecutableConfig,
        sample_size: int,
    ) -> Tuple[CostBreakdown, float]:
        """小样本快速评估。"""
        # 临时修改评估配置
        original_sample_size = self._evaluator.eval_config.sample_size
        self._evaluator.eval_config.sample_size = sample_size
        
        result = self._evaluator.evaluate(cfg)
        
        # 恢复原始配置
        self._evaluator.eval_config.sample_size = original_sample_size
        return result
    
    def _full_evaluate(self, cfg: DJExecutableConfig):
        """完整评估。"""
        return self._evaluator.evaluate(cfg)
```

**集成方式**：

```python
# 在 BeamSearchStrategy 中使用
class BeamSearchStrategy:
    def search(self, root: DJExecutableConfig) -> SearchReport:
        # 创建早停评估器
        early_stop_evaluator = EarlyStopEvaluator(
            evaluator=self._evaluator,
            config=EarlyStopConfig(
                min_sample_size=10,
                early_stop_threshold=0.7,
            ),
            get_current_best=lambda: self._get_current_best(),
        )
        
        # 使用早停评估器进行搜索
        ...
    
    def _get_current_best(self) -> Optional[Tuple[CostBreakdown, float]]:
        """获取当前最优结果。"""
        if not self._all_candidates:
            return None
        best = max(self._all_candidates, key=lambda c: c.quality)
        return best.cost, best.quality
```

**优点**：
- 显著减少低质量候选的评估成本
- 实现相对简单

**注意**：
- 阈值设置需要调优
- 可能错过"后期翻盘"的候选

---

### 4. 并行评估

**原理**：同时评估多个候选配置，利用多核 CPU 或异步 IO 提高评估吞吐量。

**实现方案**：

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple
import asyncio

class ParallelEvaluator:
    """并行评估器。"""
    
    def __init__(
        self,
        evaluator,
        max_workers: int = 4,
        use_process_pool: bool = False,
    ):
        """
        Args:
            evaluator: 底层评估器
            max_workers: 最大并行数
            use_process_pool: 是否使用进程池（适用于 CPU 密集型）
        """
        self._evaluator = evaluator
        self._max_workers = max_workers
        self._use_process_pool = use_process_pool
    
    def evaluate_batch(
        self,
        configs: List[DJExecutableConfig],
    ) -> List[Tuple[CostBreakdown, float]]:
        """
        并行评估多个配置。
        
        Args:
            configs: 配置列表
        
        Returns:
            对应的 (cost, quality) 列表
        """
        if self._use_process_pool:
            return self._evaluate_with_process_pool(configs)
        else:
            return self._evaluate_with_thread_pool(configs)
    
    def _evaluate_with_thread_pool(
        self,
        configs: List[DJExecutableConfig],
    ) -> List[Tuple[CostBreakdown, float]]:
        """使用线程池并行评估。"""
        results = [None] * len(configs)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._evaluator.evaluate, cfg): i
                for i, cfg in enumerate(configs)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        return results
    
    def _evaluate_with_process_pool(
        self,
        configs: List[DJExecutableConfig],
    ) -> List[Tuple[CostBreakdown, float]]:
        """使用进程池并行评估。"""
        results = [None] * len(configs)
        
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_evaluate_in_process, self._evaluator, cfg): i
                for i, cfg in enumerate(configs)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        return results


def _evaluate_in_process(evaluator, cfg):
    """进程池工作函数。"""
    return evaluator.evaluate(cfg)


class AsyncEvaluator:
    """异步评估器（适用于 IO 密集型，如 LLM API 调用）。"""
    
    def __init__(self, evaluator, max_concurrent: int = 10):
        self._evaluator = evaluator
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_batch_async(
        self,
        configs: List[DJExecutableConfig],
    ) -> List[Tuple[CostBreakdown, float]]:
        """异步并行评估多个配置。"""
        tasks = [self._evaluate_with_semaphore(cfg) for cfg in configs]
        return await asyncio.gather(*tasks)
    
    async def _evaluate_with_semaphore(self, cfg):
        """带信号量限制的异步评估。"""
        async with self._semaphore:
            # 如果底层评估器支持异步，直接调用
            if hasattr(self._evaluator, 'evaluate_async'):
                return await self._evaluator.evaluate_async(cfg)
            # 否则在线程池中运行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._evaluator.evaluate, cfg)
```

**在 BeamSearch 中使用**：

```python
class BeamSearchStrategy:
    def _evaluate_candidates_batch(
        self,
        candidates: List[BeamCandidate],
    ) -> List[Tuple[CostBreakdown, float]]:
        """批量评估候选配置。"""
        configs = [c.config for c in candidates]
        
        parallel_evaluator = ParallelEvaluator(
            evaluator=self._evaluator,
            max_workers=self._beam_config.parallel_workers,
        )
        
        results = parallel_evaluator.evaluate_batch(configs)
        return results
```

**配置项**：

```python
class BeamSearchConfig(BaseModel):
    # ... 其他配置 ...
    
    parallel_evaluation: bool = Field(
        default=False,
        description="启用并行评估。",
    )
    
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="并行评估的工作线程/进程数。",
    )
    
    use_process_pool: bool = Field(
        default=False,
        description="使用进程池而非线程池（适用于 CPU 密集型评估）。",
    )
```

**优点**：
- 显著提高评估吞吐量
- 充分利用多核资源
- 特别适合 LLM API 调用（IO 密集型）

**注意**：
- 需要处理并发安全（如果评估器有状态）
- 进程池需要处理序列化问题
- 可能增加 LLM API 的并发压力

---

## 实施优先级建议

| 优先级 | 方案 | 复杂度 | 收益 | 状态 |
|--------|------|--------|------|------|
| **高** | 固定采样 | 低 | 高 | **已实现** |
| **高** | 缓存评估结果 | 低 | 高 | 待实施 |
| **中** | 早停 + 剪枝 | 中 | 中 | 待实施 |
| **中** | 并行评估 | 中 | 高 | 待实施 |
| **低** | 增量评估 | 高 | 高 | 待实施 |

---

## 实施路径

### 第零阶段：固定采样（已实现）

1. ✅ `EvalConfig` 新增 `fixed_samples` 字段
2. ✅ `RealPipelineEvaluator` 实现 `prepare_fixed_samples` 方法
3. ✅ `BeamSearchStrategy` 在搜索开始时调用准备方法
4. ✅ 更新文档

### 第一阶段：缓存评估结果

1. 实现 `CachedEvaluator` 包装类
2. 集成到 `BeamSearchStrategy`
3. 添加缓存统计指标到 `SearchReport.metrics`

### 第二阶段：早停 + 剪枝

1. 实现 `EarlyStopEvaluator` 和 `EarlyStopConfig`
2. 添加小样本快速评估逻辑
3. 可配置的阈值参数

### 第三阶段：并行评估

1. 实现 `ParallelEvaluator` 和 `AsyncEvaluator`
2. 修改 `BeamSearchStrategy` 支持批量评估
3. 添加相关配置项

### 第四阶段：增量评估（可选）

1. 实现 `ExecutionTrace` 和中间结果存储
2. 修改执行器支持单算子执行
3. 实现增量评估逻辑

---

## 参考资料

- docetl 的 MOAR 搜索使用小样本快速评估
- 贝叶斯优化中的早停策略
- 并行超参数优化框架（如 Optuna、Ray Tune）