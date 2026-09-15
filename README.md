# Chameleon

这是对论文《Chameleon: Adaptive Fault Tolerance for Distributed Training via Real-time Policy Selection》的功能级复现。项目实现独立 PyTorch DP + PP、真实 1F1B、数据 rerouting、非对称 dynamic topology、完整 AdamW 状态迁移，以及在 committed-step 安全点上的真实 worker kill/recovery；不复现论文硬件吞吐数字，也不实现 Megatron、TP/EP、故障探测、任意 mid-step 回滚或磁盘 checkpoint。

## 固定验收环境

最终 GPU 验收环境是 Ubuntu 24.04、Python 3.12.3（`/usr/bin/python`）、工作目录 `/workspace`、NVIDIA PyTorch 25.06（2.8.0a0+5228986）、CUDA 12.9.1、NCCL 2.27.3 和 8 张可见 GPU。环境中的依赖版本及完整约束见 [总计划](docs/MASTER_PLAN.md)。Windows CPU/Gloo 只用于额外开发验证，不能替代该容器中的 CPU/GPU 验收。

项目不自动安装、升级或降级依赖。`requirements.txt` 中的 Ruff 仅声明 Task 15 lint 工具；声明本身不执行安装。

## 算法与语义边界

Decision Center 独立构造 rerouting candidate 与 Algorithm 1 找到的 best dynamic candidate，再按论文 Equation 8 自动选择唯一计划：

```text
score = (B / t_step) * ((D - t_transition) / D)
```

`B` 是每个全局 step 固定的 global sample count。`D` 必须由调用者显式给出、有限且大于 0；Profiler 不预测 D 或 MTBF。`D <= t_transition` 的 candidate 在该窗口不可用。完全同分时采用项目规定的确定性规则：更小 transition 优先，再按稳定 plan ID。

Estimator 实现论文 Equation 9–14：Equation 9/10 表达对称 pipeline 与多 pipeline 中最慢者的 step 时间；Equation 11 使用真实 1F1B 操作依赖；Equation 12/13 估计 stage rerouting，并把 `Fi >= Ndp` 判为不可行；Equation 14 估计 layer、端点模块、activation 与 optimizer memory。估计值和 runtime 实测值在报告中分开记录。

Algorithm 1 的预计算 cache 只保存 dynamic search 结果。预计算在 committed safe point、worker 尚存活时针对假设故障完成；真实 kill 后重新检查 live survivor state，再命中对应 search。key 包含故障/拓扑状态、survivor identities 与 generation、模型和配置 identity、完整 profile hash、搜索范围与 memory capacity；`D` 不进入 key。每次选择都会重新计算 Equation 8，live source map 和迁移 manifest 也会重新构造。过期或改变的 profile、模型、配置或故障状态会产生 cache miss。

一条 sample 是固定长度序列：先对该 sample 的有效 token loss 求平均，再对 global samples 求和；所有参数的 owner 梯度做 AllReduce SUM 后只除一次 global sample count。下一步 sample IDs 仅由 `committed_global_step * B` 决定，不因 rerouting 或 topology rebuild 跳过、重放或额外创建样本。

恢复覆盖当前模型中全部 `requires_grad=True` 参数，包括 embedding、所有 blocks、最终 LayerNorm、LM head，以及每个参数对应的 AdamW `step`、`exp_avg`、`exp_avg_sq`。唯一来源是 survivor memory；controller 不持有完整模型备份，最后一个完整副本丢失会在提交新 topology 前抛出 `UnrecoverableStateError`。

kill 只允许发生在所有 worker 已完成 optimizer step、controller 已提交 global step 且没有在途计算/通信的安全点。本项目不声称支持任意 mid-step kill 或事务回滚。

## CLI

仓库提供 [CPU 配置](configs/tiny_cpu.json) 和 [8-GPU 配置](configs/tiny_8gpu.json)。所有配置、profile、报告和 artifact 路径均相对当前项目工作目录。

生成带真实执行时间、tensor memory 和两进程 P2P/bootstrap calibration 的 profile：

```bash
python -m chameleon profile --config configs/tiny_cpu.json --output artifacts/profiles/tiny-cpu.json
```

运行真实 DP/PP training workers：

```bash
python -m chameleon train --config configs/tiny_cpu.json --steps 3
```

在三个 committed steps 后先于安全点预计算假设故障的 dynamic search，再真实 kill 一个 worker、用显式 D 执行 Equation 8，并运行选中的恢复路径：

```bash
python -m chameleon recover-demo \
  --config configs/tiny_cpu.json \
  --profile artifacts/profiles/tiny-cpu.json \
  --inter-fault-duration-s 1000
```

`recover-demo` 没有默认 D、MTBF 推断或 `force-policy` 参数。JSON 报告包含版本、seed、sample IDs/count、B/D、两个 candidates 的 step/transition/score、选中策略、source map、kill PID/exit code、state hashes、数值容差、实际 topology/timing 和资源清理结果；执行期失败也会写入 `status: failed` 和错误信息。

将配置换成 `configs/tiny_8gpu.json` 即使用 8 张真实 GPU 与 NCCL；GPU 不足或 NCCL 不可用会直接失败，不会 skip、mock 或 fallback 到 CPU。

## 测试与最终 runner

Task 15 的直接 CPU 检查：

```bash
python -m pytest tests/integration/test_cli.py tests/integration/test_plan_cache.py -q --device cpu
python -m pytest tests/unit -q --device cpu
python scripts/run_cpu_e2e.py
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
```

固定服务器的最终 GPU 检查：

```bash
python -m pytest tests/unit -q --device cuda --world-size 1 --require-gpu
python scripts/run_gpu_e2e.py --world-size 8 --backend nccl
```

两个 runner 共用一个 matrix 实现，直接调度已有的 unit/integration/distributed/e2e pytest 文件。CPU 按场景使用真实 2/4/5/6/7 worker；GPU 按场景使用真实 1/2/4/5/7/8 device。每个阶段分别写 JUnit、终端 log，并汇总为 JSON；任何失败、error 或 skip 都会记录 `failed_stage` 并使整体失败。GPU runner 先严格检查 8-GPU 固定容器合同，失败时立即停止，绝不回退 CPU。

项目的真实执行状态只记录在 [docs/PROGRESS.md](docs/PROGRESS.md)。在所有服务器 GPU 必测通过前，不应把最终验收标记为完成；有限测试也不构成对任意模型或任意故障永久正确的证明。
