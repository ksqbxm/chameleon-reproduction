# Task 06：Algorithm 1 与 batch/layer search

前置：05。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 递归枚举 survivor 数分成 dp 个 pipeline lengths，全部落在显式 Rpp，dp 落在 Rdp；去重并稳定排序。
2. batch 按节点比例预分配；递归枚举剩余分配方案，交给时间估计比较。
3. 每个分配出现 0 时，从当前最大且>1的 partition 转移1并重复；并列用 pipeline ID；总数守恒。
4. layers 均分后枚举 remainder stages；所有 pipeline 包含完整模型，端点附属模块计入成本。
5. 通过 Estimator 过滤 OOM，Algorithm 1 只返回 best dynamic plan，稳定同时间 tie-break。
6. 分别为每个可能故障数/实际 survivor 状态搜索，不能在不同资源数候选之间混选一个 runtime plan。

## 必须运行

```powershell
python -m pytest tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu
```

## 测试场景与完成标准

- 小空间全枚举 oracle 对照候选集和最佳 dynamic 时间。
- 单/多个初始 zero partition、反复 donor、donor 并列；可修复零值不得直接淘汰。
- Nm<dp 才不可满足，修复后每项>=1且 sum=global Nm；少余数与多个余数分配覆盖。
- layer remainder、端点 memory 导致 OOM、所有 plan OOM、无合法整数分拆。
- 明确断言候选中没有 rerouting，接口不使用 D、不完成 policy selection。
