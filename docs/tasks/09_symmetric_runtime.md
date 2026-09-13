# Task 09：真实对称 DP/PP 与 1F1B

前置：08。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. spawn 真实 worker，每 worker 持有本地 stage、所有本地 trainable parameters和AdamW状态。
2. 建立 dense rank与稳定worker ID映射，DP groups、PP P2P和1F1B执行。
3. activation/gradient真实send/recv；端点处理tokens/labels；warmup/steady/cooldown均记录trace。
4. 同参数owners AllReduce SUM，按global samples统一归一化；记录global loss sum与提交确认。
5. 全worker提交后推进committed step，测试harness可以在安全点暂停；所有退出路径清理group/PID。
6. 将真实runtime trace接入Profiler/Estimator并报告估计与实测，不要求论文性能误差。

## 必须运行

```powershell
python -m pytest tests/distributed/test_symmetric_training.py tests/integration/test_runtime_profile.py -q --device cpu --world-size 4
```

服务器：

```bash
python -m pytest tests/distributed/test_symmetric_training.py tests/integration/test_runtime_profile.py -q --device cuda --world-size 4 --require-gpu
```

## 测试场景与完成标准

- DP2/PP2至少4个独立PID；FP64数值对照至少3 steps，另GPU FP32 smoke。
- embedding、blocks、final norm、head参数/梯度/AdamW全部对照reference，不只loss。
- 单micro-batch、少于PP深度和常规1F1B均无死锁，真实trace依赖正确。
- global IDs不重不漏、sum/count正确、committed step确认后才增加。
- GPU使用真实NCCL/P2P，不转CPU完成训练/AllReduce。
