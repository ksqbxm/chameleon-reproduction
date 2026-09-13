# Task 08：Equation 8 adaptive policy selector

前置：07。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 独立构造 rerouting candidate；用 Eq. 12/13估计时间、保留布局、均匀 stage task routing。
2. 调用 Algorithm 1 获得 best dynamic candidate，再调用 Restorer 估计其 transition。
3. evaluate_candidates 返回两候选的时间/内存/可行性，不自行选 policy。
4. select 接收 inter_fault_duration_s=D，严格执行 (B/t_step)*(D-t_transition)/D，自动选 score 最大的唯一 plan。
5. 校验有限数、时间符号、D<=transition、无候选；完全同分采用总计划 tie-break。
6. 输出完整决策推导；无 D 禁止恢复，不添加 MTBF/默认 policy/step-only fallback。

## 必须运行

```powershell
python -m pytest tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu
```

## 测试场景与完成标准

- 独立 Oracle 不调用生产 scorer。示例 B=10、rerouting(step=2,transition=0)、dynamic(step=1,transition=10)，break-even D=20。
- D=15：rerouting 胜；D=40：dynamic 胜；两者 dynamic step 均更小，证明不是 min(step)。
- 同分、单候选、OOM、丢副本、D<=transition、非法/缺失 D 均覆盖。
- 组合测试真正调用 Planner、Estimator、Restorer，不用一个伪造 DecisionResult 跳过组合。
- Algorithm 1 的 best dynamic 不随 D 改变，最终选中 policy 随 D 改变。
- 受控 profiling fixture 允许用于数学测试，真实训练执行的策略切换必须在13验证。
