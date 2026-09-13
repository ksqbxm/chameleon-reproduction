# Task 15：CLI、预计算缓存与完整验收

前置：14。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 提供profile/train/recover-demo CLI和tiny CPU/8GPU JSON配置；recover-demo要求显式D且不预测MTBF。
2. 预计算允许1..k故障场景的dynamic search，cache按状态/model/config/profile键控；D改变重新Equation8评分。
3. 统一CPU/GPU测试runner，调用同一测试/生产模块，不另写第二套恢复逻辑。
4. 建立scripts/run_gpu_e2e.py --world-size 8 --backend nccl；输出JSON/JUnit和失败阶段。
5. README说明运行环境、论文公式、未知D来源、safe-point kill边界、所有trainable恢复和global samples语义。
6. 检查所有旧路径/孤儿逻辑、异常fallback、mockGPU、skipGPU；只保留当前实现。

## 必须运行

```powershell
python -m pytest tests/integration/test_cli.py tests/integration/test_plan_cache.py -q --device cpu
python -m pytest tests/unit -q --device cpu
python scripts/run_cpu_e2e.py
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
```

服务器：

```bash
python -m pytest tests/unit -q --device cuda --world-size 1 --require-gpu
python scripts/run_gpu_e2e.py --world-size 8 --backend nccl
```

CPU runner必须执行全部integration/distributed/e2e，并按对应task需要的2/4/5/6/7个真实进程调度，不能统一按world-size=1运行。GPU runner必须执行全部integration/distributed/e2e，并按各task需要的1/2/4/5/7/8个真实device场景执行，不只跑一个8rank smoke。显式8GPU环境检查失败即停止，不允许CPU fallback；各子suite无GPU skip。runner调用上文及01-14的同一测试文件，逐命令收集退出码，任何子suite失败即整体失败。

## 最终通过标准

- 覆盖01-14所有CPU/GPU必测和组合；短D rerouting、长D dynamic都是真实kill后实际执行。
- [5,3,2]以及不等sample sizes的dynamic/rerouting gradients、loss、parameters、AdamW都与reference一致。
- 全部trainable恢复、连续kill、最后副本丢失错误、generation/source identity与资源清理正确。
- 缓存hit/miss、过期profile、D切换无旧决策污染；实际CLI执行并输出完整记录。
- 报告包含版本、seed、sample IDs/count、B/D、候选times/scores、selected policy、source map、PID/exitcode、state hashes、容差和pass/fail。
- 任何GPU必测未运行、失败或skip均不能标最终完成；必须记录实测失败并修复回归。
- PROGRESS记录真实命令/退出码；不宣称有限测试保证任意模型/任意故障正确。
