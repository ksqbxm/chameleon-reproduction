# Task 13：单故障真实kill与adaptive policy E2E

前置：12。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 运行初始对称训练至少3 steps，等待完整step_committed安全点。
2. 真正Process.kill；Linux SIGKILL、Windows TerminateProcess；join确认退出后harness提交event。
3. 调用真实Planner/Estimator/Restorer生成两类候选，Equation8选policy，无force-policy参数。
4. 为可重复功能测试使用受控profile输入正常Estimator流程，使rerouting慢step/低transition与dynamic快step/高transition形成trade-off；不得伪造选择结果。
5. 同一故障状态分别使用D_short/D_long运行两个独立case，实际执行不同恢复路径。
6. 恢复所有parameter/AdamW，继续训练并对照不中断reference，记录estimated与actual耗时的区别。

## 必须运行

```powershell
python -m pytest tests/e2e/test_kill_adaptive_policy.py -q --device cpu --world-size 6
```

服务器：

```bash
python -m pytest tests/e2e/test_kill_adaptive_policy.py -q --device cuda --world-size 8 --require-gpu
```

## 测试场景与完成标准

- CPU DP3/PP2，GPU DP4/PP2；每个case真实kill一个stage worker，存活DP state完整。
- 从候选值独立计算break-even D，选严格两侧且dynamic能恢复的短/长D。
- 短D选择rerouting：布局保持、peer实际额外执行任务；长D选择dynamic：布局实际改变、P2P迁移真实执行。
- 最终都对照single-process参数、loss、gradient、AdamW state，并核对sample IDs与committed step。
- 确认动态step虽更小，短D仍选rerouting；不能使用min(step)或默认策略。
- 日志至少包含B/D/t_step/t_transition/score/selected policy、PID exitcode、state sources及hash、实际新topology。
- 未测GPU或仅selector unit pass不算完成本task。
