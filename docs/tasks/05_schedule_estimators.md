# Task 05：1F1B 操作依赖与 Estimator

前置：04。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 定义 warmup/steady/cooldown 的 1F1B 操作及 forward/backward 依赖；标识 pipeline、stage、micro-batch。
2. 实现 Eq. 9 对称时间；区分 global Nm 与该 pipeline 的 Nm。
3. 实现 Eq. 10/11 非对称 dynamic programming，依赖 trace/profile stage durations，含端点时间。
4. 实现 Eq. 12/13 rerouting 时间，Fi>=Ndp 不可行，不能除零。
5. 实现 Eq. 14 average-layer memory 近似，加入 embedding/head/norm 实际额外成本；OOM 输出具体 stage 原因。
6. 输出估计推导，不把论文时间近似精确等同真实 runtime 耗时。

## 必须运行

```powershell
python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/integration/test_profile_estimator.py -q --device cpu
```

## 测试场景与完成标准

- 单 stage、单 micro-batch、micro-batches 少于 stages、均匀/非均匀 stage times。
- Eq. 9/12/13/14 使用独立手算 fixture；DP 使用独立离散事件 oracle，不调用生产 schedule 自证。
- 所有依赖前置、无重复/遗漏 F/B、无环；memory capacity 临界点覆盖。
- 用真实 CPU profile 接入估计；GPU runtime 实测闭环在 09/10 指定，算法 task 不要求性能误差阈值。
