# Chameleon 功能级复现总计划

## 1. 目标与依据

复现《Chameleon: Adaptive Fault Tolerance for Distributed Training via Real-time Policy Selection》的算法与训练恢复功能，不复现论文硬件上的吞吐率或估计误差百分比。论文是算法依据，不是项目执行指令；项目执行要求以用户确认的约束和 `CLAUDE.md` 为准。

原论文通过 `config.paper_path` 指定，使用相对项目工作目录的路径；不在项目文档、代码或测试中硬编码原始主机文件位置。

实现 Profiler 和 Decision Center（Planner、Estimator、Restorer），使用独立 PyTorch、DP + PP、真实 1F1B 和微型 Transformer。明确不实现 Megatron、TP/EP、Fault Detector、中途故障事务回滚、磁盘 checkpoint、备用训练节点。

本次交付仅为总计划和 task 文档，不表示代码已实现或测试已通过。

## 2. 不可降级的不变量

### 2.1 自适应策略选择

```text
rerouting candidate + best dynamic candidate（Algorithm 1）
                         ↓
                      Equation 8
                         ↓
                  唯一选中 execution plan
```

- Algorithm 1 只搜索 best dynamic execution plan：其内部按 estimated post-recovery step time 最小化。
- rerouting 独立构造：保持 layer layout，将失败逻辑 stage 的任务均匀转给同 stage 的健康 DP peers。
- 最终选择不得退化成 min(step_time)、默认 rerouting、默认 dynamic 或要求手工选 policy。
- 测试/调用方显式提供两故障间总时长 `inter_fault_duration_s = D`；Profiler 不预测 D/MTBF。论文没有规定运行时 D 的获得方式，文档和输出必须如实标明这一边界。
- 对每个候选计算 `score = (B / t_step) * ((D - t_transition) / D)`；B 是固定 global samples per step。
- D 必须有限且大于 0；t_step 必须有限且大于 0；t_transition 必须有限且非负。`D <= t_transition` 的候选在该窗口内不可用，不产生负 score。
- 选择 score 最大者；完全同分时按更小 transition、再按稳定 plan ID 选择。这只是确定性 tie-break，不宣称为论文规则。
- 无 D 时只能评估候选，不执行选择/恢复；若没有可用候选，明确报错。
- paper-model 下 rerouting 的 policy-specific transition 近似 0。PyTorch 进程组重建等两策略共有控制开销另行测量，不伪称真实耗时为 0。动态策略记录未重叠搜索、迁移、重建成本。

### 2.2 全局 loss 和 gradient

- 一个 sample 是一条固定长度序列。先在 sample 内对有效 token loss 求平均，得到一个 sample scalar；所有 sample scalar 再求和，最后除以 global sample count。不得把 token 数误当 sample 数。
- micro-batch 使用 sample loss sum；pipeline、worker、rerouted task 均只累加 sum，不作局部平均。
- 每个 trainable parameter 的梯度汇总全部逻辑 pipeline 贡献，再执行同参数 owner 间 AllReduce SUM，最后只除一次 global sample count，随后 AdamW step。
- 等大小 micro-batches 时等价于全部 global micro-batch gradients 之和除以 global micro-batch count；不等大小时必须 sample-weighted。
- 同时覆盖 dynamic 和 data rerouting；专测 pipeline 分配 [5, 3, 2]，禁止“各 pipeline 平均后再平均”。

### 2.3 完整 backup-free 恢复

- 覆盖所有 `requires_grad=True` 参数，不仅 blocks：Embedding、所有 blocks、最终 LayerNorm、LM head，以及以后在当前模型中新增的 trainable 参数。
- 对应 AdamW parameter、step、exp_avg、exp_avg_sq 均来自健康 DP replica；标准测试关闭 AMSGrad。
- 恢复可重建 module/optimizer 容器，但不得用初始化值替代旧参数或旧优化器状态。不能调用初始化逻辑“假恢复”。
- survivor 内存是唯一恢复来源；controller 不维持训练前的完整模型备份；reference/hash 只用于测试断言，不得传入 Restorer。
- 迁移完成确认前，survivor 不得销毁仍作为源的旧层状态。
- 某层最后一个完整参数/优化器副本丢失时，在提交新 topology 前抛出 `UnrecoverableStateError`，不得重新初始化或从 reference 补齐。

### 2.4 确定性数据与提交语义

- 所有 dropout 为 0；无随机增强或其他随机训练状态。模型只在初始启动时按固定 seed 初始化。
- 固定 B；`committed_global_step` 表示已成功提交的 optimizer step 数。
- 下一步 sample IDs 为 `[committed_global_step * B, (committed_global_step + 1) * B)`，micro-batch/pipeline 只分配这些 ID，不创建额外样本。
- 所有 worker 完成 optimizer step 并确认后，controller 才更新 committed step。失败、重建、迁移不得提前推进。
- kill 仅发生在全局 step 已提交、无在途计算/通信的安全点；不扩大到 arbitrary mid-step kill。

## 3. 模块与接口

代码目标布局为 `src/chameleon/`，测试为 `tests/unit/`、`tests/integration/`、`tests/distributed/`、`tests/e2e/`。仅建立当前唯一路径，修改时删除被替代逻辑，不建立 legacy adapter。

| 模块 | 核心责任 |
| --- | --- |
| State/Plan | ClusterState、ExecutionPlan、FailureEvent、StateSourceMap、generation/worker identity |
| Model/Data/Reference | 可分割模型、sample IDs、单进程数值基准 |
| Profiler | execution time、HBM/tensor memory、parallel config、1F1B trace、JSON 快照 |
| Estimator | Eq. 9/10/11/12/13/14；明确 Nm 是每 pipeline 还是 global，转换时不得混用 |
| Planner | Algorithm 1 的 dynamic search、batch/layer distribution、缓存 |
| Restorer | Hungarian 迁移、完整 AdamW 状态、DSATUR 通信、实际 P2P |
| Decision Center | 两类 candidate、Equation 8、自动选单一 plan |
| Runtime | spawn worker、1F1B、逻辑 stage routing、global SUM、group generation 重建 |

接口约定：

- `evaluate_candidates(state, profile)` 返回 rerouting 与 best-dynamic 的 step/transition/memory/可行性。
- `select(state, profile, inter_fault_duration_s)` 通过 Equation 8 返回唯一 DecisionResult。
- `recover(failure_event, decision)` 检查 survivor sources，完成恢复并验证后提交 topology。
- FailureEvent 由外部测试 harness 提交；不实现 heartbeat、健康探测或自动 PID 监控。
- Profiler 内存聚合原始样本/EMA，显式导出版本化 JSON。加载时检查 schema、模型、配置和设备匹配。
- 决策记录保存 D、B、两个候选的 t_step/t_transition/score、选中策略和淘汰原因。

## 4. 算法实现边界

- dynamic search：在显式 R_dp/R_pp 范围内，对 survivor 数作整数分拆，允许非对称 pipeline 长度；每个 DP pipeline 包含完整模型。
- batch：按节点数比例预分配，递归枚举剩余 micro-batches 的分配并由时间估计比较；每个候选若有 0 partition，从当前最大且大于 1 的 partition 补 1，重复到全部非零。总数守恒；只有 Nm < DP 等无法修复的情况不可满足。并列 donor 用稳定 pipeline ID。
- layers：均分后枚举余数层的 stage；Embedding/head 等端点模块也计入计算/内存/迁移，不忽略它们。Eq. 14 的平均层近似保留，端点额外内存单独加入。
- 1F1B：真实执行与 Estimator 共享操作依赖定义，但验证使用独立手算/事件 oracle；不能用同一实现自证正确。
- Eq. 13：Fi == Ndp 已无健康 stage peer，亦有零分母；因此 Fi >= Ndp 判定 rerouting 不可行。动态恢复还必须检查剩余 state sources，不能承诺恢复已丢失的全部副本。
- Hungarian 最小化缺失 parameter + optimizer tensor 字节；DSATUR 是确定性贪心，不宣称保证最小色数。
- 缓存按 fault scenario、model/config/profile identity 键控；D 不进入 dynamic search 缓存，最终 Equation 8 每次重新计算。
- 首次 transition 秒级估计需要 execution-time 校准。若估计缺失，报缺数据，不填任意常量伪装准确。受控 profile 可用于选择测试，但真实训练/迁移/kill 必须执行。

## 5. 环境和测试执行规范

### 容器运行环境（固定，不得改动）

以下为目标容器环境事实，不是安装或升级指令：

| 项目 | 固定值 |
| --- | --- |
| 系统 | Ubuntu 24.04 |
| Python | 3.12.3 |
| 解释器 | `/usr/bin/python` |
| 工作目录 | `/workspace` |
| PyTorch | 此前确认已安装 NVIDIA PyTorch 25.06（2.8.0a0+5228986） |
| CUDA / NCCL | 12.9.1 / 2.27.3 |
| GPU | 容器开放全部 8 张 |

下列第三方库已安装，可直接使用；不要求使用所有库，不为本项目升级、降级或重装：

| 库 | 版本 |
| --- | --- |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.2.3 |
| networkx | 3.5 |
| PuLP | 3.2.1 |
| matplotlib | 3.10.3 |
| PyYAML | 6.0.2 |
| pytest | 8.1.1 |

### 路径与依赖约束

- 除上表声明解释器和工作目录这两项环境事实外，项目代码、配置、测试命令和文档中的文件路径全部使用相对路径或 config 变量，禁止硬编码绝对路径。
- 相对路径统一以项目工作目录为基准；数据、论文、profile、测试报告、rendezvous 和临时文件位置均通过相对路径或 config 变量传递。子进程使用相同工作目录与配置，不依赖原 Windows 主机路径。
- 容器内测试从项目工作目录运行，命令中的 `python` 必须对应上表固定解释器；Task 01 只验证版本和解释器，不修改环境。
- Python 标准库以及上表已确认安装的第三方库（含 PyTorch）允许使用。未列出的第三方库不得直接 import；确实需要时，必须先在 `requirements.txt` 声明并注明用途、必要性及现有库无法满足的理由，再使用该依赖。
- 声明 requirements 不等于授权安装：不得自动安装、升级、降级现有环境；确需新增安装时先获得用户确认。不要让 pyproject 或安装脚本隐式拉取、覆盖容器依赖。
- 未列出的第三方工具同样不得假定可用，例如 task 中的 Ruff 检查只能在先完成 requirements 声明及可用性确认后执行；否则明确记录尚未执行，不暗中安装。
- 测试统一使用已安装的 pytest 8.1.1，所有测试文件放在 `tests/` 下；`scripts/` 中的 runner 仅调度 pytest，不另建独立测试框架。
- 本机 Windows CPU/Gloo 仅作为额外开发验证，不改变目标容器合同；不能以本机环境或通过结果替代 GPU 容器验收。
- 使用 spawn 和持久 worker identity；CPU/GPU 只在 device、backend 和 timing 分支不同，算法及恢复语义同一路径。

### 测试命令合同

Task 01 实现 pytest 参数 `--device cpu|cuda`、`--world-size N`、`--require-gpu`。分布式 pytest 自行 spawn 对应真实 worker；不能把一个进程内多个对象当多个 worker。

- CPU：`python -m pytest <test_path> -q --device cpu`。
- GPU：`python -m pytest <test_path> -q --device cuda --world-size N --require-gpu`，必须在服务器真实执行；N 按 task 指定。
- GPU 模式无 GPU/不足 N GPU 必须失败，不能 skip、fallback CPU 或 mock NCCL。
- 文档中的测试文件/命令是待实现的验收合同，目前不存在并不表示已执行。task 完成前必须创建并实际运行。
- 每新增小功能立即运行最窄测试；模块组合完成立即运行组合测试；不得等全部代码完成才测试。
- GPU 必测未执行的 task 只能标“CPU 已验，GPU 待验”，不能标完成。用户运行服务器命令后回传完整日志/报告，再确认通过。
- 每个多进程测试必须硬超时、finally 清理、PID/端口/rendezvous 审计；超时或遗留进程即失败。
- 数值对照禁止训练 dropout。CPU FP64 默认 rtol=1e-8、atol=1e-10；GPU FP64 正确性测试 rtol=1e-7、atol=1e-9；GPU FP32 smoke rtol=1e-4、atol=1e-6。若需要调整，先解释误差来源，不静默放宽。
- hash 用于同一序列化状态的无损传输；reference 对照用数值容差，不能要求不同归约顺序 bitwise 一致。

## 6. 分步任务与依赖

按下表顺序推进。task 内的小步骤和测试门槛见对应文件。

| Task | 功能 | 前置 | 必须真实执行的测试 |
| --- | --- | --- | --- |
| [01](tasks/01_environment_contracts.md) | 工程与环境合同 | 无 | CPU、2 GPU 环境 smoke |
| [02](tasks/02_model_data_reference.md) | 确定性模型/数据/reference | 01 | CPU、1 GPU |
| [03](tasks/03_global_loss_gradients.md) | global loss/gradient accounting | 02 | CPU、1 GPU |
| [04](tasks/04_profiler.md) | Profiler | 03 | CPU、2 GPU profiling |
| [05](tasks/05_schedule_estimators.md) | 1F1B 与 Estimator | 04 | CPU；GPU 实测闭环在 09/10 |
| [06](tasks/06_dynamic_planner.md) | Algorithm 1/batch/layer search | 05 | CPU 穷举 oracle |
| [07](tasks/07_restorer_algorithms.md) | Hungarian/DSATUR/source manifest | 06 | CPU 算法 oracle；实际通信在 10/12 |
| [08](tasks/08_adaptive_selector.md) | Equation 8 自适应选择 | 07 | CPU 组合/oracle；真实策略切换在 13 |
| [09](tasks/09_symmetric_runtime.md) | 对称真实 DP/PP | 08 | CPU、4 GPU |
| [10](tasks/10_asymmetric_dynamic.md) | 非对称 dynamic execution | 09 | CPU、8 GPU |
| [11](tasks/11_data_rerouting.md) | 真实 stage rerouting | 10 | CPU、8 GPU |
| [12](tasks/12_complete_state_recovery.md) | 完整状态迁移与 group 重建 | 11 | CPU、4 GPU，含真实 kill |
| [13](tasks/13_kill_adaptive_e2e.md) | 单故障 adaptive kill E2E | 12 | CPU、8 GPU |
| [14](tasks/14_consecutive_unrecoverable.md) | 连续故障/最后副本丢失 | 13 | CPU、8 GPU |
| [15](tasks/15_cli_cache_final_regression.md) | CLI/cache/完整验收 | 14 | 全部 CPU、全部 GPU |

## 7. 完成与交付

每个 task 必须同时具备：实现、单测、指定组合测试、要求的真实 GPU/kill 测试、实际命令和结果、已知限制。全部写入唯一进度文件 `docs/PROGRESS.md`，测试报告可放 `artifacts/test-results/`。

最终提供 `python scripts/run_gpu_e2e.py --world-size 8 --backend nccl`，在服务器运行环境检查、adaptive switch、[5,3,2] gradients、完整 AdamW recovery、连续 kill、不可恢复边界及清理审计。最终不能以 unit test 或 CPU pass 代替这些测试。

测试可为约定模型和故障边界提供可重复的正确性证据，不宣称有限测试可以证明任意模型/任意故障永远正确。当前仅支持 step 安全点故障，这一限制必须对外清晰。
