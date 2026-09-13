# Task 01：工程骨架与环境合同

前置：无。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 建立 src package、pyproject、pytest 和最小开发依赖；torch 由环境提供，不覆盖服务器 nightly。
2. 定义 ModelConfig、ClusterState、ExecutionPlan、FailureEvent、DecisionResult 和 UnrecoverableStateError；固定 B、dropout=0、AdamW AMSGrad=False。
3. 建立 pytest 参数 --device、--world-size、--require-gpu，以及 CPU/Gloo、CUDA/NCCL fixtures。
4. 实现真实 spawn 环境 smoke、硬超时、finally 清理和进程/端口审计。GPU fixture 不允许 skip/fallback。

每个步骤完成立即写并运行对应单测，先验证非法配置再验证有效配置。

## 必须运行

```powershell
python -m pytest tests/unit/test_contracts.py -q --device cpu
python -m pytest tests/integration/test_environment.py -q --device cpu --world-size 2
```

服务器：

```bash
python -m pytest tests/integration/test_environment.py -q --device cuda --world-size 2 --require-gpu
```

## 测试场景与完成标准

- 参数错误、非有限时间、非法 D、模型配置非法均有清晰异常。
- 两个不同 PID 分别拥有不同 CUDA device，真实 NCCL SUM 返回预期结果。
- 输出完整容器版本和全部 8 GPU 可见性；使用 2 GPU 不代表只检查 2 GPU 可见性。
- 子进程超时/异常后无遗留。CPU 和服务器 smoke 均通过后才完成。
- 命令、版本、退出码和清理结果写入 PROGRESS，不创建根目录进度文件。
