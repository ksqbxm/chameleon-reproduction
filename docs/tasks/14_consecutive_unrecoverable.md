# Task 14：连续故障与最后副本丢失

前置：13。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 第一次kill→恢复→至少2个新committed steps，再对当前topology真实kill第二worker。
2. 每次由更新的survivor/profile/state sources重新搜索；不能复用旧rank/source或过期decision。
3. 通过外部D短/长与候选实际分数验证rerouting→dynamic顺序，策略由Equation8选择，不强制policy。
4. 独立场景逐步kill同一trainable参数全部健康副本，最后一次必须UnrecoverableStateError。
5. 错误路径检查topology提交边界、optimizer未重置、无controller/reference备份救场。
6. 成功/异常/超时最终清理全部workers、groups、ports，输出资源审计。

## 必须运行

```powershell
python -m pytest tests/e2e/test_consecutive_kills.py tests/e2e/test_unrecoverable_state.py tests/e2e/test_resource_cleanup.py -q --device cpu --world-size 6
```

服务器：

```bash
python -m pytest tests/e2e/test_consecutive_kills.py tests/e2e/test_unrecoverable_state.py tests/e2e/test_resource_cleanup.py -q --device cuda --world-size 8 --require-gpu
```

## 测试场景与完成标准

- 每次kill均确认目标PID退出；第二次故障发生在真正恢复并训练之后。
- 旧generation/rank不能误当当前source；survivor count和层副本分布精确更新。
- 全程sample IDs无跳过/重放，parameter与AdamW对照不中断reference。
- 单独覆盖blocks、embedding、final norm、head最后副本丢失；必须全参数inventory检查。
- error必须发生于新topology提交前；禁止任何初始化/磁盘/reference来源。
- 短timeout stress重复运行至少3轮，无deadlock、遗留PID或端口。
