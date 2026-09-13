# Task 12：完整模型/AdamW状态恢复与新group

前置：11。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 在至少3个已提交AdamW steps后暂停于安全点，状态非零且每参数step可审计。
2. 建立survivor-only控制通道；稳定worker identity不因dense rank变化丢失。
3. 真实kill一个worker，确认exitcode/PID已死，由harness显式提交FailureEvent。
4. survivors销毁旧group并加入新的generation，保留旧本地状态，不能因group异常全部退出后重初始化。
5. 检查完整StateSourceMap，Hungarian映射后仅P2P传输缺失parameter/AdamW tensors；保留本地intersection。
6. target完成验证ACK后才释放不再需要的source tensors并提交topology。
7. 模块/optimizer容器可重建，但逐参数旧state必须加载；恢复完成后下一step继续。

## 必须运行

```powershell
python -m pytest tests/distributed/test_full_state_transfer.py tests/e2e/test_kill_group_rebuild.py -q --device cpu --world-size 4
```

服务器：

```bash
python -m pytest tests/distributed/test_full_state_transfer.py tests/e2e/test_kill_group_rebuild.py -q --device cuda --world-size 4 --require-gpu
```

## 测试场景与完成标准

- DP2/PP2初始4 workers真实kill后3 survivors，group重建与状态传输真实发生。
- embedding/blocks/final norm/head的parameter、step、exp_avg、exp_avg_sq恢复前后hash一致。
- 测试reference/hash只供断言；Restorer不能读取reference或磁盘checkpoint。
- spy初始化路径和传输source audit，证明不是重置optimizer或用测试state补缺。
- source旧layer即使target重新分区也不能在ACK前销毁。
- 部分manifest缺tensor、端点唯一source缺失、transfer异常不得提交半个topology。
- 新topology至少继续2 steps对照不中断reference；杀死worker确实未重生成为备份。
- group重建若hang/失败，必须修真实runtime，不能mock destroy/init或skip NCCL。
