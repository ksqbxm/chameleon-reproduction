# Task 10：非对称 dynamic execution与通信

前置：09。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 执行不同pipeline lengths与layers partitions；每pipeline包含完整模型，不遗漏端点。
2. 建立每trainable参数的owner group，所有rank按一致顺序创建groups。
3. 实际按DSATUR rounds发起async AllReduce，同轮无device冲突，round之间等待完成。
4. 执行不均匀batch [5,3,2]；local accumulation只SUM，global SUM后按样本数除一次。
5. 将修复后的zero partition plan从Planner接入Runtime，证明不只是纸面修复。
6. 记录各pipeline完成时间/1F1B trace，Estimator使用max pipeline而非均值。

## 必须运行

```powershell
python -m pytest tests/distributed/test_asymmetric_training.py tests/distributed/test_colored_allreduce.py tests/integration/test_planner_runtime.py -q --device cpu --world-size 7
```

服务器：

```bash
python -m pytest tests/distributed/test_asymmetric_training.py tests/distributed/test_colored_allreduce.py tests/integration/test_planner_runtime.py -q --device cuda --world-size 8 --require-gpu
```

## 测试场景与完成标准

- CPU pipeline lengths [2,2,3]；GPU [2,3,3]，batch [5,3,2]，global Nm=10。
- 不等sample sizes和partial micro-batch；每层gradient/parameters/AdamW至少3 steps对照single-process。
- “pipeline means再平均”的错误结果必须与reference不同，生产结果必须正确。
- 每层/端点实际同步groups与color rounds可审计，无layer漏同步。
- Profiler→Estimator→Planner→Restorer scheduling→Runtime组合实际运行。
- 本task运行新建非对称topology，不宣称已完成故障恢复；完整迁移在12。
