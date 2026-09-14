# 项目进度

本文件是唯一进度记录位置。Task 01 的工程骨架、数据合同和环境测试已实现；Task 02 的模型、确定性数据、单进程 reference 和 step commit 已实现；Task 03 的全局 loss/sample accounting、单设备 owner gradient SUM 与一次归一化及对照测试已实现；Task 04 的 Profiler 和校准已实现，服务器回归及双 GPU 状态见下表；Task 05 的 1F1B 依赖、Eq.9–14 估计和 profile 接口已实现，审阅后 161 项算法单测通过，3 项真实 profile 接入因本机缺少 torch 待验；Task 06 的 dynamic Planner、batch/layer search 已实现，审阅后本机 CPU 指定组合 288 passed，固定容器复验待执行；Task 07 的完整 survivor sources、Hungarian 字节匹配、DSATUR 和 Restorer manifest/transition 规划已实现，审阅后指定算法 CPU 组合 135 passed、相关算法回归 584 passed，补充真实 inventory 与字节校准因本机缺 torch 待验。当前 profile 唯一格式为 version 3，旧文件需要重新采样。本机结果不替代固定容器验收。真实 GPU 验收由用户在服务器启动；此前按用户要求重跑本机 CUDA 命令，因缺少 torch 在配置阶段退出。未实现分布式训练或实际状态迁移恢复，未执行真实 GPU 或训练进程 kill 测试。

Task 08 的 Equation8 自适应选择、独立 rerouting candidate 与 Planner/Estimator/Restorer 组合已实现并审阅修正；本机指定 CPU 组合 108 passed，合同及 Task05–08 算法回归 746 passed。固定容器复验待执行；受影响真实 inventory/校准补测为 39 passed / 22 errors（缺 torch），真实训练中的策略切换按总计划在 Task13 验证。

## 状态

| Task | 状态 | 实际测试 |
| --- | --- | --- |
| 01 | 已修正 NCCL 元数据判断；非 torch 合同已验；真实 CPU/GPU 待验 | 最新全量回归中 84 passed / 13 errors；9 项新增元数据测试及 4 项真实 smoke 因缺少 torch 报错；GPU 未执行 |
| 02 | 已修正 optimizer 完成与 commit 的时序；非 torch 单测已验；CPU/GPU 数值待验 | 最新全量回归中 17 passed / 15 errors；训练/模型测试因缺少 torch 报错；GPU 未执行 |
| 03 | owner 清单/重叠检查已实现；CUDA 浮点断言已修正；待服务器重跑验收 | 用户回传 CUDA 70 passed / 1 failed；修正后本机 CPU 32 passed / 39 errors，CUDA 配置阶段退出，均因缺少 torch |
| 04 | 服务器全仓回归通过；NGC 版本表示检查已修正；双 GPU 校准待重跑 | 用户回传全仓 318 passed、Task04 CUDA 115 passed / 3 setup errors（环境合同阻断）；本次合同测试 54 passed；原 CUDA 命令本机因缺 torch 配置阶段退出 |
| 05 | 算法单测已验；真实 CPU profile 接入待验 | 审阅后 Task05 161 passed / 3 errors（缺 torch）；全仓 421 passed / 89 errors（缺 torch）；GPU runtime 闭环在 09/10 |
| 06 | 已实现并审阅修正；本机 CPU oracle 已验；固定容器待复验 | 审阅后指定四文件 288 passed；调度/Estimator 回归 161 passed；无失败或跳过；本 task 不要求独立 GPU 测试 |
| 07 | 已审阅并修正 plan/estimate/ACK 绑定与 transition 计费；指定 CPU oracle 已验；固定容器与真实路径待验 | 指定四文件 135 passed；Task05/06/07 算法组合 584 passed；真实 inventory/校准/profile 回归 101 passed / 41 errors（缺 torch）；实际通信在 10/12 |
| 08 | 已实现并审阅修正；本机 CPU 算法组合已验；固定容器与真实路径待验 | 指定两文件 108 passed；合同及 Task05–08 算法回归 746 passed；真实 inventory/校准补测 39 passed / 22 errors（缺 torch）；真实策略切换在 13 |
| 09-15 | 待实施 | 未执行 |

GPU 必测未执行时，不得将对应 task 标为完成。

## 计划文档交付检查（2026-09-13）

- 已创建总计划、15个task文档和本进度文件，共17份Markdown。
- 实际运行只读文档检查：task数量=15、全部相对链接可解析、每个task具有“必须运行”测试命令段；检查通过。
- 复查最终runner按各task的进程/device规模分别执行，禁止统一world-size=1替代分布式组合。
- 此次没有实现算法/训练代码，没有运行CPU训练、GPU或kill测试。01-15均仍待实施。

## 容器合同补充

- 总计划已加入固定 Ubuntu 24.04 / Python 3.12.3、解释器、工作目录和已装第三方库版本。
- 已加入相对路径/config 变量、禁止改动容器环境、新依赖先声明理由、pytest 测试统一放 tests/ 的约束。
- 移除总计划中硬编码的原 Windows 论文路径，改为 config.paper_path。
- 本次仅修改文档，未改动容器、安装依赖或运行训练/GPU 测试。

## Task 01 实现与开发验证（2026-09-13）

- 修改范围：`pyproject.toml`、`requirements.txt`、`.gitignore`、`src/chameleon/`、`tests/conftest.py`、`tests/unit/`、`tests/integration/test_environment.py` 和本进度文件。仓库未发现额外 AGENTS.md；实现遵循用户提供的全局规则及 `CLAUDE.md`。
- 工程骨架：src package，可直接通过 pytest 的 pythonpath 运行；项目运行依赖列表为空，torch 和 pytest 由既有容器提供。setuptools 仅作为构建依赖声明用途，不执行安装或隔离构建。
- 数据合同：不可变 ModelConfig、WorkerIdentity、ClusterState、ExecutionPlan、FailureEvent、DecisionResult 和 UnrecoverableStateError；校验正整数、worker ID/rank/generation、已提交 step、固定 global batch、dropout=0、AMSGrad=False、有限时间与有效 D。micro-batch 不要求整除 B，以保留总计划规定的不等大小批次。
- 环境与通信：pytest 提供 `--device cpu|cuda`、`--world-size N`、`--require-gpu` 及 CPU/Gloo、CUDA/NCCL fixtures；GPU 不足或 NCCL 不可用直接失败。真实 spawn workers 绑定不同 CUDA device，进行 FP64 AllReduce SUM。正常执行、注入 worker 异常和硬超时分别测试；父进程 finally 中 terminate/join/kill，检查 PID/exitcode、rendezvous 目录删除、TCP 端口无监听且可重用。
- 固定容器检查：Ubuntu 24.04、Python 3.12.3、声明的解释器/工作目录、torch nightly、全部已列第三方库、NCCL 2.27.3 和完整 8 GPU 清单。torch 的 CUDA 字段只暴露 12.9，补丁版本通过 NVIDIA 容器的 CUDA_VERSION 元数据核验为 12.9.1；缺少元数据时明确报无法验证，不推断补丁版本。
- 测试环境：额外开发验证使用 Windows 11、Miniforge base Python 3.13.12、pytest 9.1.1；torch 不存在，device=cpu，未创建分布式 backend。该本机结果不替代合同规定的 Python 3.12.3 / pytest 8.1.1 容器验收。按用户补充要求，不访问服务器、不安装或改变环境。

实际运行记录：

| 小功能 / 命令 | 退出码 | 结果 |
| --- | --- | --- |
| 初版配置合同：`python -m pytest tests/unit/test_contracts.py -q --device cpu` | 0 | 47 passed |
| 环境输入/版本合同：`python -m pytest tests/unit/test_environment_contract.py -q --device cpu` | 0 | 27 passed |
| 加入 CLI 参数测试后的组合单测 | 0 | 77 passed |
| 最终组合：`python -m pytest tests/unit -q --device cpu` | 0 | 84 passed，0 failed，0 skipped；包含新增 cluster/failure counter 边界测试 |
| `python -m pytest tests/integration/test_environment.py --collect-only -q --device cpu --world-size 2` | 0 | 4 tests collected；只验证可发现性，不代表执行通过 |
| `python -m pytest tests/integration/test_environment.py -q --device cpu --world-size 2` | 1 | 4 errors：ModuleNotFoundError: No module named 'torch'；发生在 fixture 中，worker 尚未启动 |
| 带 JUnit 输出的同一 CPU 集成命令 | 1 | 同样 4 errors；完整日志与 XML 已保存 |
| `python -m compileall -q src tests` | 0 | Python 语法检查通过 |
| 标准库 tomllib 解析 pyproject 并检查 src 测试路径 / 空运行依赖 | 0 | 通过 |

- 报告/日志：`artifacts/test-results/task01-unit.log`、`task01-unit.xml`、`task01-local-summary.json`、`task01-environment-cpu.log`、`task01-environment-cpu.xml`。这些是实际本机报告，artifacts 已加入忽略规则。
- 一次本机 JUnit 输出阶段出现 Windows native exception 诊断（pytest 的 platform.node 调用栈）；随后无 JUnit 的最终命令正常退出 0，84 passed。未修改或修补本机环境。
- PID/端口/rendezvous 清理：本机集成测试因缺少 torch 未 spawn；正常通信、异常与硬超时后的实际清理审计均未验证。服务器执行会产生 `smoke-{device}-{normal|error|hang}.json`，记录真实 worker PID、exitcode、SUM、错误与清理结果，并在 pytest 终端摘要中输出本次报告。
- 未验证项：固定容器完整版本核验、真实双进程 CPU/Gloo、双 GPU/NCCL SUM、真实异常/超时清理。GPU 命令未执行，不能标 Task 01 完成。
- 下一项依赖：Task 02 仍依赖 Task 01 的服务器验收；本次未提前实现 Task 02。
- 最终检查：逐文件复核新增代码、依赖、路径和本进度记录；未引入 skip、CPU fallback、mock NCCL 或安装操作。当前目录不是 Git 仓库，`git status --short` 无法执行，未声称完成 Git diff 检查。实际报告 XML 可解析，单测 XML 为 84 tests / 0 errors / 0 failures / 0 skipped，CPU 集成 XML 为 4 tests / 4 errors / 0 skipped。

服务器验收命令（从项目工作目录运行，python 对应总计划指定解释器；无需安装项目）：

```bash
python -m pytest tests/unit -q --device cpu --junitxml=artifacts/test-results/task01-server-unit.xml
python -m pytest tests/integration/test_environment.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task01-server-cpu.xml
python -m pytest tests/integration/test_environment.py -q --device cuda --world-size 2 --require-gpu --junitxml=artifacts/test-results/task01-server-gpu.xml
```

用户回传上述实际完整日志/报告后，再确认 CPU/GPU 与资源清理是否通过并更新验收状态。

## Task 02 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task 01/02/03 文档及已有工程。没有发现额外仓库 AGENTS.md。本次按用户明确要求推进 Task 02 实现，Task 01 的服务器环境验收仍待回传。
- 修改范围仅为新增 `src/chameleon/model.py`、`data.py`、`reference.py`、`step.py`，新增 `tests/unit/test_model_data.py`、`test_reference.py`，以及本进度文件。没有修改依赖、环境配置、Task 01 代码或后续 task 文档。
- 模型：Embedding → 因果 Transformer blocks → final LayerNorm → 独立 LM head；使用已有 PyTorch 模块，所有模块和 attention dropout 为 0。模块以 `embedding`、`blocks.N`、`final_norm`、`lm_head` 的稳定名称暴露，便于以后切分。inventory 从当前全部 `requires_grad=True` named parameters 生成，包含名称、模块身份、shape、numel 和 tensor 字节，覆盖端点模块及以后新增的 trainable 模块，并排除冻结参数。
- 初始化：`build_initial_model` 仅作为首次启动入口，在隔离的 CPU RNG 状态中按 config.seed 初始化，再转换到指定 device/dtype。reference 接收已建好的 model，不在训练或 step commit 中初始化或重新设 seed。当前未实现 Restorer，不提供用初始化代替恢复的路径。
- 数据：固定 config 中的 B 和序列长度；`next_sample_ids` 仅从 ClusterState.committed_global_step 与 B 计算区间。`make_batch` 使用 sample ID/位置的整数函数生成固定长度的 next-token inputs/targets，不依赖 RNG、cursor、micro-batch 划分或 generation。
- reference：独立执行整个 global batch，不依赖待实现的分布式逻辑。每个 sample 内求有效 token loss mean，再对 sample scalars 求 sum；对 sum backward 后每个参数梯度只除一次 B，然后执行 AdamW（AMSGrad=False）。每步返回完整 trainable 参数、归一化梯度和按稳定参数名索引的 step/exp_avg/exp_avg_sq 独立 CPU tensor 副本，并记录 IDs、sample losses、global sum/count/mean 与提交后的 step。reference 状态仅用于测试对照，不提供给恢复模块。
- 提交：`StepCommit` 检查参与者完整 worker identity/rank/generation 和下一步编号，拒绝重复、旧 step 与超前确认。部分确认不改变 immutable ClusterState；最后一个确认才推进 committed step。reference 的唯一参与者在 optimizer.step 成功返回后确认；optimizer 更新前抛错不会推进数据。该测试只覆盖更新前异常，不扩展为中途 optimizer 事务回滚。
- 新测试覆盖精确 inventory、端点和新增/冻结参数、未绑定权重、dropout 与因果性、初始 seed 和训练 RNG 不漂移、固定 ID 区间与重建、不重不漏的连续 steps、数据重排确定性、非法 IDs/确认、有效 token sample mean、至少 3 步的完整非默认 AdamW 状态与可重复性。另以独立 log_softmax 损失及手写 AdamW 更新公式对照 loss、全部 gradients、parameters 与两个 moments，避免仅用同一训练实现自证。数值容差遵循 CPU FP64 rtol=1e-8/atol=1e-10、GPU FP64 rtol=1e-7/atol=1e-9。
- 本机环境：Windows、Python 3.13.12、pytest 9.1.1；默认 Python、两套已有 conda 环境及 bundled runtime 均没有 torch，WSL 列表访问报 E_ACCESSDENIED。没有安装、升级、降级依赖或访问服务器。以上环境不是目标容器合同，也不能替代目标 Python 3.12.3 / pytest 8.1.1 的验收。

实际运行记录（小功能完成后立即运行最窄测试）：

| 小功能 / 命令 | 退出码 | 实际结果 |
| --- | --- | --- |
| 初版模型：`python -m pytest tests/unit/test_model_data.py -q --device cpu` | 1 | 6 errors，全部在 torch fixture 导入阶段 |
| 加入数据后的同一命令 | 1 | 5 passed / 7 errors / 0 skipped，错误仍为缺少 torch |
| 初版 reference/commit：`python -m pytest tests/unit/test_reference.py -q --device cpu` | 1 | 12 passed / 3 errors / 0 skipped，错误仍为缺少 torch |
| 加入独立 AdamW oracle、提交时序和输入边界后的同一命令 | 1 | 12 passed / 6 errors / 0 skipped，错误仍为缺少 torch |
| 必需 CPU 组合：`python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cpu --junitxml=artifacts/test-results/task02-cpu.xml` | 1 | 17 passed / 13 errors / 0 skipped；30 tests，13 项训练/模型 fixture 无法导入 torch |
| 完整 unit 回归：`python -m pytest tests/unit -q --device cpu --junitxml=artifacts/test-results/task02-unit-regression.xml` | 1 | 101 passed / 13 errors / 0 skipped；114 tests，仍为同一批缺少 torch 的 fixture 错误 |
| 既有合同回归：`python -m pytest tests/unit/test_contracts.py tests/unit/test_environment_contract.py tests/unit/test_pytest_options.py -q --device cpu --junitxml=artifacts/test-results/task02-existing-unit.xml` | 0 | 84 passed / 0 failed / 0 skipped |
| 非 torch 新功能：`python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cpu -k 'ids_use_only or invalid_sample_ids or commit_requires or confirmation or failure_and_topology' --junitxml=artifacts/test-results/task02-accounting.xml` | 0 | 17 passed / 13 deselected / 0 failed / 0 skipped；仅验证 ID 与提交逻辑，不能视为训练通过 |
| `python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py --collect-only -q --device cpu` | 0 | 30 tests collected；仅验证可发现性 |
| `python -m compileall -q src tests` | 0 | 语法检查通过；不代表 torch API 和数值正确性已执行通过 |
| 用户补充要求之前的一次本机 CUDA 命令（Task 02 指定的 world-size 1 / require-gpu） | 4 | pytest 配置阶段报 No module named 'torch'；没有运行测试、启动 GPU 或生成 XML，仅保存失败日志。用户随后明确要求 GPU 测试交由服务器运行，本机未再尝试 |

- 日志及 JUnit：`artifacts/test-results/task02-cpu.log/.xml`、`task02-unit-regression.log/.xml`、`task02-existing-unit.log/.xml`、`task02-accounting.log/.xml`；CUDA 环境失败日志为 `task02-gpu-unavailable.log`，不存在同名 XML。环境和 XML 摘要为 `task02-local-summary.json`。
- 实际 XML 已使用标准库解析核对：CPU 组合 30 tests / 13 errors，完整 unit 114 tests / 13 errors，既有合同 84 tests / 0 errors，非 torch 新功能 17 tests / 0 errors；四份报告均 0 failures / 0 skipped。
- 最终检查：复核全部新增文件、依赖与相对路径；未引入未声明第三方库、skip、CPU fallback、mock NCCL、安装操作或额外进度文档。目录没有 Git 元数据，不能运行 Git diff，改为逐文件检查本次新增代码与本进度修改。Task 02 是单进程测试，没有 worker PID/端口/rendezvous 资源，不宣称执行了分布式清理或 kill 审计。
- 未验证项：模型真实 forward/backward、CPU/GPU FP64 数值、3 步 AdamW 状态和独立 oracle、训练 RNG 行为，均需要有 torch 的环境执行；GPU 由用户按以下指令在服务器启动。当前不标为验收完成，也不标“CPU 已验，GPU 待验”，因为 CPU 训练同样尚未通过。
- 下一项依赖：Task 03 使用本次独立 reference 对照 global loss/gradient accounting；本次未实现 Task 03。

服务器启动指令（从项目工作目录运行，使用总计划指定的既有 python；无需安装项目或依赖）：

```bash
python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cpu --junitxml=artifacts/test-results/task02-server-cpu.xml
python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cuda --world-size 1 --require-gpu --junitxml=artifacts/test-results/task02-server-gpu.xml
```

第二条是 Task 02 的单 GPU 验收。没有真实 GPU、NCCL 不可用或环境不满足时，应直接报错，不 skip 或切换 CPU。用户回传完整日志/报告后再确认数值验收结果；按用户要求，本机不执行 GPU 测试。

## Task 03 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task 03、Task 10/11 的验收边界、既有模型/data/reference/step 代码和测试。适用父目录及仓库未发现额外 AGENTS.md，遵循用户提供的全局 AGENTS.md。按用户要求推进实现，Task 01/02 的服务器验收仍待回传。
- 修改范围：新增 `src/chameleon/global_loss.py`、`tests/unit/test_global_loss.py`、`tests/integration/test_partitioned_gradient_oracle.py`，更新本进度文件。没有修改 reference、依赖或容器配置，没有新增安装操作。
- loss：`micro_batch_loss_sum` 对每个 sample 的有效 token cross-entropy 求 mean，再对 samples 求 SUM；忽略 target=-100，拒绝无有效 token 的 sample 和空/不匹配维度。独立手算 loss 与 log_softmax/autograd oracle 检查 loss、gradient 和 masked token 的零梯度。
- accounting：`GlobalBatchAccounting` 每个 global step 创建一次，依据 immutable ClusterState 的 committed step 和固定 B 构造期望 IDs。逐 micro-batch 核验实际 sample_count 与 IDs 长度，拒绝重复、丢失、旧/未来 step IDs 和非法 count/loss。完整分区才返回 `loss_global_sum`、`global_sample_count`、`loss_global_mean`，不平均 pipeline means；非法批次不消耗 IDs。
- gradient：各 owner 先在自身 `.grad` 中累加所有原生/rerouted logical task 的 SUM；按稳定参数名对所有 owner buffers 再求 SUM，最后只除一次实际 global sample count，将结果复制给相同参数的 owners。不同 stages 可提供不同参数名，包括端点。拒绝重复 buffer、缺少 tensor、shape/dtype/device 不一致、空 owner 集合、分区不完整和重复归一化；验证 buffers 后才修改梯度。
- 共用语义：实现没有 policy 分支。组合测试分别构造 dynamic 的三个数学 owner，以及 rerouting 中一个 peer 为两条逻辑 pipeline 共用同一梯度 buffer 的情况；都调用相同 accounting。它们是单设备数学 replicas，仅有一个 step commit participant，不作为多个真实 worker 或 stage-level rerouting 的证明。实际 AllReduce、DP/PP 和 P2P 必须在 Task 09/10/11 另行实现验收。
- 测试场景：显式 10 个 micro-batches 按 [5,3,2] 分配，等大小场景含 10 samples；不等大小/partial 场景的 sizes 为 `((2,1,3,2,1),(4,1,2),(3,2))`，含 21 samples，禁止除以 micro-batch 数 10。纯数学测试验证 loss=94/10 与 pipeline 均值的均值不同。两种 policy、两种 sample-size 场景各连续 3 步对照独立全局 reference，比较全部 trainable 参数、归一化梯度和完整 AdamW step/exp_avg/exp_avg_sq，并包含错误平均 pipeline gradients/loss 的负对照。使用非默认 lr/betas/eps/weight_decay，AMSGrad=False，dropout=0。FP64 容差遵循 CPU rtol=1e-8/atol=1e-10、GPU rtol=1e-7/atol=1e-9，没有放宽。
- 本机环境重新只读确认：Windows、Python 3.13.12、pytest 9.1.1，当前 Python 无 torch。未访问服务器、安装/升级/降级依赖或执行 GPU 命令。本机结果不替代目标容器合同。

实际运行记录（完成小功能后先执行最窄测试）：

| 小功能 / 命令 | 退出码 | 实际结果 |
| --- | --- | --- |
| ID/count：`python -m pytest tests/unit/test_global_loss.py -q --device cpu -k 'not micro_batch_loss and not loss_rejects and not owner_sum and not bad_owner' --junitxml=artifacts/test-results/task03-accounting.xml` | 0 | 17 passed / 11 deselected / 0 skipped；仅验证无需 torch 的 accounting |
| loss/gradient 单元：`python -m pytest tests/unit/test_global_loss.py -q --device cpu --junitxml=artifacts/test-results/task03-global-loss-unit.xml` | 1 | 17 passed / 11 errors / 0 skipped；全部错误发生在 torch fixture 导入阶段 |
| 3 步组合：`python -m pytest tests/integration/test_partitioned_gradient_oracle.py -q --device cpu --junitxml=artifacts/test-results/task03-partitioned-oracle.xml` | 1 | 4 errors / 0 skipped；两种 policy × 两种 size 场景均无法导入 torch |
| Task 03 必需 CPU 组合：`python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cpu --junitxml=artifacts/test-results/task03-cpu.xml` | 1 | 17 passed / 15 errors / 0 failures / 0 skipped；ModuleNotFoundError: No module named 'torch' |
| 完整 unit 回归：`python -m pytest tests/unit -q --device cpu --junitxml=artifacts/test-results/task03-unit-regression.xml` | 1 | 118 passed / 24 errors / 0 failures / 0 skipped；新增 11 项及既有 13 项 torch fixture 错误 |
| `python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py --collect-only -q --device cpu` | 0 | 32 tests collected；仅验证可发现性 |
| `python -m compileall -q src tests` | 0 | Python 语法检查通过；不代表 torch API 或数值对照通过 |
| 标准库解析本次 5 份 JUnit XML | 0 | 报告可解析，统计与终端一致，全部 0 failures / 0 skipped；摘要写入 task03-local-summary.json |

- 日志/XML：`artifacts/test-results/task03-accounting.log/.xml`、`task03-global-loss-unit.log/.xml`、`task03-partitioned-oracle.log/.xml`、`task03-cpu.log/.xml`、`task03-unit-regression.log/.xml`；collection 日志为 `task03-collection.log`，环境/报告摘要为 `task03-local-summary.json`。
- 最终检查：逐文件复核新增实现/测试和进度更新；所有新增 import 均为标准库、既有 package 或允许的 torch/pytest，文件位置使用相对路径，无 skip、mock NCCL、CPU fallback 或额外进度文档。当前目录不是 Git 仓库，`git status --short` 报 not a git repository，无法运行 Git diff，不宣称 diff 检查通过。本 task 没有 spawn、PID/端口/rendezvous 或 kill，不宣称多进程清理审计通过。
- 未验证项：真实有效-token loss/backward、owner gradient SUM/一次归一化、重复归一化与坏 tensor 边界、两种 policy 的 3 步 AdamW 对照均待有 torch 的环境执行；GPU 由用户启动服务器命令。Task 03 不标完成，也不标“CPU 已验，GPU 待验”，因为 CPU 数值尚未通过。
- 下一项依赖：Task 04 的 Profiler 使用本次 sample loss/gradient 语义；本次未实现 Task 04。

服务器启动指令（从项目工作目录运行，python 对应总计划指定的既有解释器；无需安装项目或依赖）：

```bash
python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cpu --junitxml=artifacts/test-results/task03-server-cpu.xml
python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cuda --world-size 1 --require-gpu --junitxml=artifacts/test-results/task03-server-gpu.xml
```

第二条为 Task 03 的真实单 GPU 数学验收；无 GPU/不足 GPU/NCCL 不可用必须报错，不 skip 或 fallback。用户回传完整日志/报告后再确认验收结果。

## Task 01/02/03 审阅问题修正（2026-09-13）

- 遵循 `CLAUDE.md`、总计划和三个 task 的约束，按用户要求修正此前审阅问题。修改仅涉及 `src/chameleon/environment.py`、`reference.py`、`global_loss.py`，对应三个单元测试文件、`tests/integration/test_partitioned_gradient_oracle.py` 和本进度文件。未修改依赖、容器或后续 task。
- Task 01：NCCL 版本查询依据 distributed/NCCL 的编译可用性，不再将 CUDA/GPU 可见性当作 NCCL 支持。无 NCCL 的 CUDA build 可以报告元数据并使用 CPU/Gloo；CUDA 验收仍严格要求真实 NCCL。支持 NCCL 但无可见 GPU 时仍报告编译版本。移除清理中的 `gc.collect()`，保留原有明确的进程、进程组、端口和临时目录清理。
- Task 02：将原本位于 commit 后的 loss scalar `.item()` 移到 AdamW 更新后、commit 前；当前单设备、同一 CUDA stream 的 scalar 主机读取会等待已排队的更新完成，读取失败不会提前推进 step。没有额外同步分支、备份或 optimizer 回滚路径。更新后故障测试只验证没有 commit，丢弃已变更的 trainer，不宣称其可以重试；原有更新前故障重试仍有独立测试。
- Task 03：构造函数必须提供物理 owner ID → trainable 参数名集合的 `expected_owners`，在 step 开始时复制成固定清单。归一化只接受 owner ID → named gradients 的映射，必须与清单逐 owner 完全一致；任何漏 owner、漏参数、全体漏端点参数、额外或错拼参数都会在访问 tensors 前失败。组合测试从各 model 的 `parameter_inventory` 生成预期清单，独立于实际 `.grad` 收集。旧匿名 owner 列表接口和默认清单已完全删除，所有调用点已迁移，没有兼容适配器。
- Task 03：删除基于 `id(gradient)` 的去重，改为核验梯度布局并按 device 检查实际占用的字节区间，覆盖同对象、detach、view、transpose、部分重叠、跨参数名和跨 dtype 别名。唯一支持路径是实浮点、dense 且内部不重叠的 strided buffers；支持连续张量、dense transpose/permute、任意 singleton stride、scalar、empty tensors，以及同一 flat allocation 中互不相交的 dense views。带孔洞、内部重叠或 sparse 的人工布局明确拒绝，没有通用 stride 求交或复制兜底路径。
- Task 03：清单、tensor 类型、布局、shape/dtype/device 和所有字节区间先统一校验，再逐参数 SUM、除实际 sample count 一次、复制到各 owner。删除全模型 `sums` 临时副本字典，只保留当前参数的一个 SUM 临时 tensor；共享存储中不同参数的不相交区域仍可以正确逐参数处理。无 trainable 参数的个别 owner 可以有空清单，全部 owner 均无参数则拒绝。
- 新增 50 个参数化测试实例：Task 01 的编译支持/可见 GPU 矩阵及 CPU/Gloo 与 CUDA/NCCL 分支 9 项，Task 02 的完成读取时序/异常 2 项，Task 03 的清单、失败后修正、存储别名和合法/非法布局等 39 项。总计 200 项可收集测试。保留有效 token loss 独立 oracle、全部参数/梯度/AdamW moments 对照、两种 policy × [5,3,2]/不等样本数 × 连续 3 步，以及错误平均的负对照；没有放宽 FP64 容差。
- Task 01 元数据单测仅控制 feature flags 和查询元数据，不创建或替换 backend/collective；它们不代表真实 GPU 验收。Task 02 CUDA 时序测试记录真实 CUDA Event 并在 commit 前检查完成；数学组合测试在其全部 optimizer 更新完成后才确认单参与者 step，不代表真实多 worker 训练或恢复。
- 本机仍为 Windows / Python 3.13.12 / pytest 9.1.1，无 torch。未安装或调整环境、未访问服务器、未执行 CUDA 命令。下列错误均发生在导入 fixture 中，无法据此确认 CPU/GPU tensor API 或数值正确性。

实际运行记录：

| 小功能 / 命令 | 退出码 | 实际结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_environment_contract.py -q --device cpu --junitxml=artifacts/test-results/task01-fix-environment-unit.xml` | 1 | 27 passed / 9 errors；新增元数据 tests 无法导入 torch |
| `python -m pytest tests/unit/test_reference.py -q --device cpu --junitxml=artifacts/test-results/task02-fix-reference-unit.xml` | 1 | 12 passed / 8 errors；训练 tests 无法导入 torch |
| 补齐最后 5 项清单 tensor 测试之前：`python -m pytest tests/unit/test_global_loss.py -q --device cpu --junitxml=artifacts/test-results/task03-fix-global-loss-unit.xml` | 1 | 32 passed / 30 errors；阶段性 62 tests，非最终数量 |
| 最终 Task 03 组合：`python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cpu --junitxml=artifacts/test-results/task03-fix-cpu.xml` | 1 | 32 passed / 39 errors；71 tests，全部 errors 为缺少 torch |
| 完整回归：`python -m pytest tests/unit tests/integration -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task01-03-fix-regression.xml` | 1 | 133 passed / 67 errors；200 tests，全部 errors 为缺少 torch |
| `python -m pytest tests/unit tests/integration --collect-only -q --device cpu` | 0 | 200 tests collected；只验证可发现性 |
| `python -m compileall -q src tests` | 0 | 语法检查通过；不替代 tensor API 或数值验收 |

- 上述 5 份实际 JUnit 报告均为 0 failures / 0 skipped；使用标准库 XML parser 核对统计和逐项 error，全部 error 含 `No module named 'torch'`。报告与完整日志位于 `artifacts/test-results/`，与表中 XML 同 basename 的 `.log`；collection 为 `task01-03-fix-collection.log`，报告摘要为 `task01-03-fix-local-summary.json`。
- 当前目录没有 Git 元数据，按修改前内容快照生成 `artifacts/test-results/task01-03-fix.diff` 并逐文件复核；清单旧接口、`id(gradient)`、全模型 sums 和 gc 调用均已移除，未引入 skip、mock NCCL、CPU fallback、新依赖或其他进度文档。修改前快照只作为差异审阅数据，不是可执行旧实现或兼容路径。
- 未验证项：真实 CPU/Gloo 和 GPU/NCCL 通信/清理，真实 loss/backward、owner SUM/布局处理、3 步 AdamW 数值，以及 CUDA optimizer 完成前 commit 的时序。由于没有 torch，smoke 未启动 worker，不能声称 PID/端口/rendezvous 或 kill 审计通过。三个 task 均不标验收完成。

服务器验收命令（项目工作目录、总计划指定的既有解释器；无需安装依赖）：

```bash
python -m pytest tests/unit tests/integration -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task01-03-fix-server-cpu.xml
python -m pytest tests/unit tests/integration/test_partitioned_gradient_oracle.py -q --device cuda --world-size 1 --require-gpu --junitxml=artifacts/test-results/task01-03-fix-server-gpu.xml
python -m pytest tests/integration/test_environment.py -q --device cuda --world-size 2 --require-gpu --junitxml=artifacts/test-results/task01-fix-server-environment-gpu.xml
```

第二条验证单 GPU 模型与数学对照，第三条单独验证真实双 GPU 环境与清理，不以单 GPU 替代双 GPU 要求。用户回传实际完整日志/XML 后更新验收状态。

## Task 03 CUDA 标量容差测试修正（2026-09-13）

- 用户回传服务器 Task 03 CUDA 日志：71 tests，70 passed / 1 failed / 1 warning；唯一失败为 `0.7000000000000001 == 0.7` 的精确浮点比较。此为用户提供的服务器结果，不是本机执行结果。
- 仅修改 `tests/unit/test_global_loss.py::test_owner_sum_divides_once_and_includes_endpoint_names` 的三个端点标量断言，统一使用 `pytest.approx` 和已有 `tolerance` 的 rtol/atol；CPU 1e-8/1e-10、CUDA 1e-7/1e-9 保持不变。实现代码、其他测试语义和 requires-grad scalar warning 均未修改，无兼容路径或新依赖。
- 实际重跑 CPU：`python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cpu --junitxml=artifacts/test-results/task03-scalar-tolerance-cpu.xml`；退出码 1，32 passed / 39 errors / 0 failures / 0 skipped，全部 errors 为缺少 torch 的 fixture 导入错误。
- 实际重跑 CUDA：`python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cuda --world-size 1 --require-gpu --junitxml=artifacts/test-results/task03-scalar-tolerance-gpu.xml`；退出码 4，配置阶段报 `No module named 'torch'`，未运行 tests/GPU，也未生成 CUDA XML。最新用户重跑要求覆盖此前不再尝试本机 CUDA 命令的偏好；未安装或改变本机环境、未访问服务器。
- 标准库 AST 语法检查及直接 `pytest.approx` 检查成功：CPU/CUDA 两套既有容差均接受用户提供的 0.7000000000000001，同时拒绝 0.701。该检查不代表 Tensor CPU/GPU 数值测试通过。实际 CPU XML 已解析核对，日志保存为上述报告同 basename 的 `.log`；CUDA 仅有 `.log`。
- 最终复核：测试函数仅三个 `==` 标量断言替换为带原容差的 `pytest.approx`，生产实现未修改；未启动 worker，无 PID/端口/rendezvous 或 kill 审计。修复后的 CPU/GPU 数值仍待服务器执行，不将 Task 03 标记验收完成。
- 服务器重跑沿用 `task03-server-cpu.xml` / `task03-server-gpu.xml` 命令（见 Task 03 实现章节），运行对象仍为 `tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py`；CUDA 必须保留 `--world-size 1 --require-gpu`。

## Task 04 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task 04/05 和前置模型、data、reference、loss、environment 代码；未发现额外仓库 AGENTS.md，遵循用户提供的全局规则。按用户要求实现 Task 04，前置 task 的服务器验收状态保持原记录。
- 本次修改范围：新增 `src/chameleon/profiler.py`、`transfer_calibration.py`、`tests/unit/test_profiler.py`、`tests/integration/test_profile_roundtrip.py`、`tests/distributed/test_transfer_calibration.py`，以及本进度文件。保留工作区已有的 Task 03 标量容差测试和进度修改；没有修改其他实现、依赖或容器环境。
- Profiler：每个 trainable 模块采集 forward 时间；从模块真实 forward 的 autograd 图识别新增节点，用节点 pre/post hooks 累计该模块的 backward execution time，包含整数输入的 embedding backward。CPU 使用 monotonic clock，CUDA 记录真实 events 并在同步后读取。保存实际 step 时间、wall 时间、parallel configuration、原始样本和 EMA；事件和 hooks 在每次采样 finally 中释放。
- 内存：从完整 `parameter_inventory` 按模块统计 parameter、当前 gradient 和 AdamW tensor bytes，包含端点及新增 trainable 参数。必须先执行真实 warmup，使全部 AdamW step/exp_avg/exp_avg_sq materialize；不以初始化状态填充。CPU 明确记录 `logical_tensor_bytes`，HBM peak 为 null；CUDA 实测 peak allocated/reserved。
- activation：`output_activation_bytes` 是每模块当前 step 中最大的输出张量逻辑大小，原始样本仍保留每次 forward 的大小；`saved_activation_bytes` 是当前 step autograd 实际保存的非参数/非 buffer 张量的逻辑字节总量，按保存发生的模块归属统计，loss/runtime 单独列出。同一存储的重复保存可重复计数，这不是物理占用或 live activation peak；物理 HBM 使用 CUDA allocator 实测。
- trace：`train_profile_step` 真实执行单 worker、单 stage 的 F0/B0/F1/B1，保留不等大小末尾 micro-batch，梯度仅除一次 global sample count；optimizer 完成、CUDA 标量等待和 profiling 成功返回后才 commit。`Profiler.step/operation` 提供后续 runtime 的真实操作记录入口。本 task 未实现或声称执行多 stage 分布式 PP；操作依赖与多 stage runtime 仍属 Task 05/09。
- 通信校准：两个真实 spawn workers 绑定 CPU/Gloo 或不同 CUDA/NCCL device，预热后双向 send/recv 多种 FP64 tensor size 并验证内容。分别记录 execution/wall time；bootstrap 记录真实 rendezvous、init 和首次 barrier/CUDA 同步，包含 NCCL lazy communicator 初始化，不包含 group shutdown。所有原始校准样本可加入 Profiler 的 EMA 聚合。`transfer_time_s` 使用较慢端点的样本均值；未测边或大小明确报缺数据，不填任意带宽。
- 每次多进程校准有父进程硬超时，finally terminate/join/kill、进程组销毁和 PID/exitcode、TCP port、临时 rendezvous 目录审计；异常、超时或清理失败抛 `CalibrationError` 并保存审计报告。测试包含真实正常传输和硬超时清理；本机缺 torch，尚未实际 spawn 或验证这些审计。
- JSON：只在显式 `export_profile` 时写盘，`load_profile` 校验 version、字段、模型结构 hash、config hash、dtype/device/hardware/torch identity、parallel configuration、原始样本/EMA、trace 和 memory 结构。不保存参数值或 optimizer state，不作为恢复 checkpoint，不采集或预测 D/MTBF。schema fixtures 只用于结构/独立手算测试，不替代真实 profiling 或传输。
- 本机：Windows / Python 3.13.12 / pytest 9.1.1，无 torch。未安装、升级、降级依赖，未访问服务器，未执行 CUDA 命令；这些本机结果不替代目标容器合同或 GPU 验收。

实际运行记录（退出码非 0 的测试不记为验收通过）：

| 小功能 / 命令 | 退出码 | 实际结果 |
| --- | --- | --- |
| EMA 最窄单测：`python -m pytest tests/unit/test_profiler.py -q --device cpu -k 'raw_samples or invalid_ema or invalid_measurements'` | 0 | 11 passed / 7 deselected |
| 初版真实计时/内存单测：`python -m pytest tests/unit/test_profiler.py -q --device cpu` | 1 | 11 passed / 7 errors，均缺 torch；后续增加 saved activation 独立 inventory 测试 |
| 初版 JSON：`python -m pytest tests/integration/test_profile_roundtrip.py -q --device cpu` | 1 | 28 passed / 6 errors；5 项默认 pytest 临时目录访问拒绝，1 项缺 torch |
| 临时目录改为项目相对 artifacts 配置后，同一 JSON 命令 | 1 | 33 passed / 1 error；仅真实 profile fixture 缺 torch，文件读写和 JSON 校验已通过 |
| 初版双 rank 校准：`python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2` | 1 | 27 passed / 2 errors；真实 spawn fixtures 缺 torch |
| Task 04 必需 CPU 组合：`python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --junitxml=artifacts/test-results/task04-cpu.xml` | 1 | 44 passed / 9 errors / 0 failures / 0 skipped；53 tests，全部 errors 缺 torch |
| Task 04 必需 CPU 通信：`python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task04-transfer-cpu.xml` | 1 | 27 passed / 2 errors / 0 failures / 0 skipped；29 tests，全部 errors 缺 torch |
| 最终全仓组合：`python -m pytest tests -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task04-regression.xml` | 1 | 204 passed / 78 errors / 0 failures / 0 skipped；282 tests，全部 errors 缺 torch |
| Task 04 三文件 `--collect-only -q --device cpu` | 0 | 82 tests collected；不代表执行通过 |
| `python -m compileall -q src tests`；新增文件 AST/空白检查 | 0 | 语法和空白检查通过；不代表 torch API 或数值验证通过 |

- 三份 JUnit 已通过标准库 XML parser 核对数量和逐项错误，所有 errors 均含 `No module named 'torch'`。实际日志与 XML 同 basename，位于 `artifacts/test-results/`；另有 `task04-collection.log`、`task04-local-summary.json` 和新增文件审阅 diff `task04-new-files.diff`。
- 未验证项：真实 forward/backward 和 CUDA events、独立 parameter/gradient/AdamW/saved activation inventory、profiling 前后数值一致性、重复采样 HBM 无泄漏、真实双 rank P2P/bootstrap 和超时清理，均待有 torch 的环境执行。当前不能标“CPU 已验，GPU 待验”，也不能标 Task 04 完成。
- 下一项依赖：Task 05 读取本 task 的真实模块时间、逻辑内存及 trace；本次没有提前实现 Estimator 或 schedule。

服务器验收命令（从项目工作目录运行，使用总计划指定的既有 python，无需安装项目或依赖）：

```bash
python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --junitxml=artifacts/test-results/task04-server-cpu.xml
python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task04-server-transfer-cpu.xml
python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/distributed/test_transfer_calibration.py -q --device cuda --world-size 2 --require-gpu --junitxml=artifacts/test-results/task04-server-gpu.xml
```

第三条必须真实双 GPU/NCCL，设备不足或环境不匹配直接失败；用户回传完整日志/XML 与校准审计后，再确认 CPU/GPU 验收状态。

## Task 04 审阅与优化（2026-09-13）

- 按用户要求审查正确性、完备性和简洁性，重新核对 `CLAUDE.md`、总计划、Task 04 及相关代码。修改仅涉及两个 task04 实现、三个对应测试文件和本进度文件；保留工作区已有 Task 03 修改，没有修改依赖或环境。
- 实际先写回归测试并复现 18 个失败实例：初次 trace/step/parameter 校验 3 failed，通信 schema/输入 9 failed / 26 passed，补充 module timing/CUDA peak 2 failed，CPU HBM metrics 2 failed，rank/PID 非标量输入 2 failed / 19 passed。修正后这些非 torch 回归测试全部通过；新增真实 torch/worker 回归仍因环境缺 torch 未执行到断言。
- 正确性：旧 trace 仅检查 F/B 配对，允许单 stage F0/F1/B0/B1。现在逐 pipeline/stage 检查真实 1F1B warmup/steady/cooldown 次序、FIFO micro-batch 和 trace 时间边界；warmup 为 min(pp_size-stage-1, Nm)。拒绝 trace 超出实测 step 时间，以及模块执行时间总和超出其真实 operation 时段。独立手写 oracle 覆盖 PP=2、PP=4 且 Nm 少于 stages 的 phase 校验，不宣称已执行多 stage PP runtime。
- 实际操作关联：forward/backward 必须处于正确 scope，每个 operation 必须执行该 worker 的全部 trainable 模块；拒绝嵌套 scope、重复操作和空 backward。非参数 autograd 节点绑定创建它的真实 forward micro-batch，backward 标签必须匹配；可跨 micro-batch 复用的 AccumulateGrad 仍计时，但不能作为当前 forward 图已经 backward 的证据。删除旧 `_backward_id` 状态，仅保留当前 operation 状态和实际模块执行清单。
- 完备性：profile 唯一格式升级为 version 2，identity 加入模块 parameter bytes 清单，加载时核对全部模块及其 parameter bytes、forward/backward/activation/optimizer 原始样本数量、memory 与原始样本一致性，以及 CUDA peak 样本；CPU 禁止出现 HBM metrics。模型 hash 纳入模块 `extra_repr`，可检测例如实际 LayerNorm eps 改变；设备检查覆盖 buffers，未指定 index 的 CUDA device 显式解析为当前设备。version 1 文件直接拒绝，需要重新导出，没有旧格式解析或迁移分支。
- 资源生命周期：hook 注册和初始时间戳创建均移入同一个 try/finally，逐项登记 handles，途中失败也会移除已注册 hooks，并清空当前 operation/events 状态。测试覆盖 hook 注册途中失败、初始时间戳失败、后续重新采样和嵌套 scope 拒绝后外层仍能完成；本机尚未实际执行这些 torch 路径。
- 校准：共享唯一 tensor size 校验，先核验整数/FP64 字节边界再去重，避免非法嵌套数据触发 TypeError；lookup 不再把 bool/float 当作 rank 或字节 identity。rank/PID 的字段和整数校验移到聚合/去重前；核验 bootstrap 是 list、worker PID/exitcode/布尔状态及两个 rank 的执行环境一致性。父进程在完成判断前检查硬 deadline，避免超时后完成的 worker 被记为成功；保留现有 terminate/join/kill 和 PID/port/rendezvous 审计。新增真实 spawn 的 worker 异常测试，worker 在通信前抛错，不替换 collective/P2P。
- 简洁性：删除被替代的 scope 状态和宽松校验；复用 size 校验，合并重复的 CPU/CUDA event 测试 setup/body，设备差异只留在类型/同步断言中。没有新增兼容 adapter、默认 timing/bandwidth、恢复初始化或额外进度文档。
- 本机仍为 Windows / Python 3.13.12 / pytest 9.1.1，无 torch；未安装或改变环境，未访问服务器，未执行 CUDA 命令。CPU/GPU FP64 数值容差保持原值，未静默放宽。

实际最终验证：

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task04-review-cpu.xml` | 1 | 61 passed / 16 errors / 0 failures / 0 skipped |
| `python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task04-review-transfer-cpu.xml` | 1 | 38 passed / 3 errors / 0 failures / 0 skipped |
| `python -m pytest tests -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task04-review-regression.xml` | 1 | 232 passed / 86 errors / 0 failures / 0 skipped；318 tests |
| Task 04 三文件 `--collect-only -q --device cpu` | 0 | 118 tests collected，比审阅前增加 36 项 |
| `python -m compileall -q src tests`；修改文件 AST/空白检查 | 0 | 通过，不代表 torch API 或数值验证通过 |

- 三份 XML 已核对：77/41/318 tests，16/3/86 errors，全部错误含 `No module named 'torch'`。实际完整日志与 XML 同 basename；另有 `artifacts/test-results/task04-review-collection.log`、`task04-review-local-summary.json` 和本次审阅 diff `task04-review.diff`。修改前快照只用于差异审阅，位于忽略的 artifacts 中，不是可执行旧实现或兼容路径。
- 当前 task04 合计 99 passed / 19 errors。真实 torch 计时、数值、inventory、CUDA HBM、双 rank 校准和异常/超时清理未验证，本机没有启动 worker，不能将 CPU/GPU 验收标通过。服务器沿用上节三条验收命令，运行当前 version 2 实现并回传完整日志/XML/审计后再更新状态。

## NGC 25.06 环境合同版本表示修正（2026-09-13）

- 用户回传服务器全仓回归 318 passed、Task04 CUDA 115 passed / 3 setup errors，三个分布式测试在 fixture 的 `validate_container()` 阶段被阻断。实际元数据为 torch `2.8.0a0+5228986c39.nv25.06`、torch CUDA `12.9`、NCCL `2.27.3`、容器 CUDA_VERSION `12.9.1.010`。本次按用户要求只修正环境合同、对应单测和本进度记录，没有修改 Task04 逻辑、fixture、依赖或设备检查。上述服务器结果来自用户回传，未在本机重复执行成功。
- torch 用唯一完整匹配规则核验 `2.8.0a0`、提交前缀 `5228986`，允许十六进制提交号展开及可选 `.nv25.06` 元数据；CUDA_VERSION 用唯一完整匹配规则核验 `12.9.1`，允许末尾一个数字构建号。拒绝错误 base/prerelease、提交前缀、NGC release、CUDA patch 及非数字/多段/多余后缀；没有旧路径 adapter 或跳过合同的开关。Ubuntu、Python、解释器、工作目录、torch CUDA `12.9`、NCCL `2.27.3`、8 GPU 可见性和所有第三方库版本检查保持原值。
- 对应独立元数据 fixture 改用用户提供的真实版本表示，确保原有系统/依赖不匹配测试不会因 torch/CUDA 表示错误而误通过。新增 27 个测试实例，覆盖 8 组等价表示的通过和元数据不被修改，以及 20 组版本/类型/后缀拒绝；其中 1 个通过实例来自原有测试。
- 修改实现前运行 `python -m pytest tests/unit/test_environment_contract.py -q --device cpu -k valid_container_check_does_not_mutate_metadata --junitxml=artifacts/test-results/task04-ngc-before.xml`，退出码 1：7 failed / 1 passed / 55 deselected，复现等价表示被拒绝。
- 修改后运行 `python -m pytest tests/unit/test_environment_contract.py -q --device cpu -k 'not nccl_report and not absent_nccl' --junitxml=artifacts/test-results/task04-ngc-contract.xml`，退出码 0：54 passed / 9 deselected。未选择的 9 项为原有 torch 元数据测试，本机缺 torch；没有修改或跳过服务器 GPU 测试。
- `python -m compileall -q src/chameleon/environment.py tests/unit/test_environment_contract.py` 和 `git diff --check` 均通过。
- 按用户要求原样重跑下列 Task04 CUDA 命令，本机退出码 1：`ERROR: No module named 'torch'`。仍为 Windows / Python 3.13.12，无 torch；配置阶段退出，没有测试执行、worker 启动或本次 JUnit 生成。实际输出记录在 `artifacts/test-results/task04-ngc-cuda-local.log`，未安装依赖、切换 CPU 或替换 NCCL。服务器更新代码后仍须原样执行这条命令，才能确认三个分布式测试和清理审计通过：

```bash
python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/distributed/test_transfer_calibration.py -q --device cuda --world-size 2 --require-gpu --junitxml=artifacts/test-results/task04-server-gpu.xml
```

## Task 05 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task 05、Task 06 接口需求、既有 Profiler/模型/合同及测试；适用父目录与仓库未发现额外 AGENTS.md。严格遵循最小实现、局部修改和唯一进度文件规则。
- 公式核对依据为论文 [arXiv v4 Estimator 原文](https://arxiv.org/html/2508.21613v4#S4.SS3)。仓库没有论文文件或配置的 paper_path；未硬编码主机论文位置。
- 修改范围：新增 `src/chameleon/schedule.py`、`src/chameleon/estimators.py`、`tests/unit/test_1f1b_schedule.py`、`tests/unit/test_estimators.py`、`tests/integration/test_profile_estimator.py`；仅修改 Profiler 的顺序校验和单 stage runner 以复用同一调度，以及本进度文件。未实施 Task 06 或后续功能，未修改依赖或环境。
- 调度：每条 pipeline/stage 输出不可变 FIFO 1F1B 队列，标识 micro-batch、warmup/steady/cooldown、同 stage 前序操作、上游 forward、本地 forward 与下游 backward 依赖。单 stage 真实训练入口使用该队列；不把估计 trace 冒充实测 trace。
- 时间：Eq.9 显式从 global micro-batches / DP 转为每 pipeline 数量，不均等整数分区必须改用非对称估计。Eq.11 以线性 DAG DP 求每个操作的 max(predecessor ends) + duration，支持逐操作不同耗时；Eq.10 取不同长度/分区 pipeline 中的最大时间。profile 接口校验唯一 version 2 格式及模型/配置/设备/并行 identity，汇总所有模块（含端点）的实测 forward/backward EMA，不填写缺失 timing 常量。
- rerouting：单故障按 Eq.12，其他分布按 Eq.13；保留原 DP/layout 和每 pipeline Nm，逐 stage 导出额外计算 slots。Fi>=Ndp 在任何除法前返回不可行及具体 stage 原因，不承诺动态策略可以恢复已丢失的副本。
- 内存：Eq.14 保留 average-layer 近似和 Npp-i 激活系数（Nm 少于 stages 时也保留论文近似），单独加入全部非 block 模块的 parameter/gradient/AdamW tensor 字节及其保存激活，loss/runtime 保存激活计入最后 stage。累计 saved activation 字节按实际 forward 次数转为每 micro-batch 成本。容量相等可行、超过容量返回逐 stage 的 static/dynamic/extra 原因。
- 推导边界：输出 equation、batch scope、stage durations、操作依赖/时间及内存成本。时间为 computation 估计，尚未计入未测量的 P2P、gradient collectives、optimizer 与控制成本；逻辑 saved-tensor 字节可能包含别名重复，不等同物理峰值 HBM。GPU runtime 实测闭环按 Task 05 合同留在 09/10，不设置性能误差阈值。
- 独立验证：手写 1F1B 顺序、独立图消除检查依赖无环；Eq.9/12/13/14 使用独立手算 fixture；Eq.11 用不读取生产 schedule/dependencies 的状态机与完成事件 heap oracle，逐操作核对 start/end。覆盖单 stage/单 micro-batch、Nm<stages、非均匀与逐 micro-batch 耗时、端点/新增模块、不可用 peer、batch 守恒、profile identity/缺失数据和内存临界点。
- 真实 profile 测试已创建：实际 AdamW warmup 后采样两步，使用含 [2,2,1] samples 的 micro-batches，JSON 导出/加载后接入估计，检查端点与容量边界。人工 schema fixture 仅用于算法单测，不替代真实训练。
- 本机环境：Windows / Python 3.13.12 / pytest 9.1.1，torch 不存在，device=cpu；不替代总计划的固定容器环境。没有安装、升级、降级依赖、访问服务器或执行 GPU 测试。

实际运行记录（小功能后立即执行最窄测试）：

| 命令 / 阶段 | 退出码 | 实际结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_1f1b_schedule.py -q --device cpu` | 0 | 45 passed |
| 初版 `python -m pytest tests/unit/test_estimators.py -q --device cpu --tb=short` | 1 | 95 passed / 1 failed；手算 fixture 的容量抄写错误，修正为计算出的 239/74/167 字节 |
| 修正 fixture 并加入单 stage 内存后的两文件单测组合 | 0 | 142 passed |
| 初版 Task05 指定三文件组合 | 1 | 142 passed / 3 errors；真实 profile fixture 导入 torch 失败 |
| 最终 `python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/integration/test_profile_estimator.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task05-cpu.xml` | 1 | 150 passed / 3 errors / 0 failures / 0 skipped；153 tests |
| `python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task05-profiler-regression.xml` | 1 | 61 passed / 16 errors / 0 failures / 0 skipped；原有真实 torch 路径未执行到断言 |
| `python -m pytest tests -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task05-regression.xml` | 1 | 409 passed / 89 errors / 0 failures / 0 skipped；498 tests |
| `python -m compileall -q src tests`；`git diff --check` | 0 | 通过；不代表未运行的 torch 路径通过 |

- 三份 JUnit 已核对计数及逐项错误：所有 errors 均为 `No module named 'torch'`。实际日志与 XML 同 basename，位于 `artifacts/test-results/`；汇总为 `task05-local-summary.json`。
- 本 task 没有分布式 spawn/端口/rendezvous/kill；真实 profile 临时目录使用 finally 清理的标准库 TemporaryDirectory。本机缺 torch，尚未创建该临时目录或启动训练，不宣称实际执行了训练资源清理。
- 最终审阅：检查全部新文件与 Profiler diff，未引入未声明第三方库、skip/fallback、旧格式 adapter、初始化恢复或额外进度文档。
- 当前不能标 Task 05 验收完成，也不能标“CPU 已验”，因为其指定真实 CPU profile 接入仍未通过。Task 04 双 GPU 校准仍待服务器重跑；按用户要求已继续实施算法，不将前置未验状态隐去。

固定容器中从项目工作目录运行以下验收命令（使用既有 python，无需安装项目或依赖）：

```bash
python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/integration/test_profile_estimator.py -q --device cpu --junitxml=artifacts/test-results/task05-server-cpu.xml
```

收到真实 CPU 完整日志/XML 后再确认 Task 05 验收状态；本 task 没有独立必测 GPU 命令，GPU runtime 闭环继续遵循 Task 09/10。

## Task 05 审阅与优化（2026-09-13）

- 按用户要求复核正确性、完备性和简洁性；遵循 `CLAUDE.md`，没有修改模型/data/reference/global loss、环境合同、依赖或后续任务。修改涉及 Task05 Estimator、相关 Profiler 采样/格式和对应测试，以及本进度文件；保留之前已实现的 Task05 调度与 runner 修改。
- 首先用四项测试复现根本问题：合法 JSON 键排序导致模型布局被拒绝；末尾清空梯度后估计峰值从 428/389 降到 328/301；[12,12,6] 字节的 micro-batches 被平均为 10 而低估峰值 12；公开接口接受缺少 F→B 依赖的任意 operation 队列。实际结果为 4 failed / 105 deselected。
- 激活采样：不再只保存整步累计字节；每个真实 forward 建立包含全部模块和 loss/runtime 的整数计数行，导出逐 forward 原始样本与 EMA，step memory 保存各模块的最大样本。Estimator 对全部实测原始样本取最大值，再在 layers 间求平均；端点逐项使用实际峰值，不因短末批或后来较小的采样擦除已测峰值。删除原累计字节除以 forward 次数的路径。
- 梯度峰值：严格使用 Eq.14 的 mg=mp，包含 blocks 和端点。删除独立 average_gradient_bytes 输入，不把快照中已释放的梯度当作训练峰值零成本；profile 的 gradient inventory 仍如实记录采样结束时实际状态。
- 模块顺序：identity 增加明确的 module_order，完整且唯一覆盖 module_parameter_bytes；布局以该序列校验，不再以 JSON 对象键顺序推断模型拓扑。
- 格式变更：profile 唯一 schema 为 version 3，采样、校验、导出、加载、Estimator 和所有 profile fixture 同步更新。version 1/2 及其他版本直接拒绝，需要重新导出真实 profile；没有旧格式 adapter 或兼容分支。通信校准自身的独立 schema 未改变。
- 调度接口：公开逐操作时间接口改为 durations + pipeline/stage/micro-batch 数，由唯一 schedule builder 构造完整依赖。删除任意队列输入及为其设置的重复/缺依赖/环检查；私有 DP 仅接收生成的合法调度，保留耗时完整性与有限性校验。逐 stage 和逐 operation 时间共用一个线性 DP 实现，全部调用处已更新。
- 完备性补充：随后另外四项测试复现非峰值样本可含小数字节，以及调用方修改 layer_modules list 使内部集合为空并除零。字节原始样本现在逐项校验为非负整数；layer inventory 与 profile 一并复制。新增历史峰值保留及 module_order 缺失/重复/非法成员的回归。
- 真实 autograd oracle 已同步改为独立记录每个 micro-batch 的完整 saved tensor 总字节，对照 Profiler 的逐模块原始样本求和；检查短末批及 step 最大值。现有 hook/event 泄漏测试同时检查新计数器释放。真实训练测试没有 mock、skip 或放宽数值容差。
- 估计边界继续保留：Eq.14 是 average-layer 近似，逻辑 saved tensor 字节可重复计别名，不等同实际峰值 HBM；Eq.9/12/13 是均匀 stage 的论文近似；GPU runtime 实测闭环仍在 Task09/10。

实际执行记录：

| 命令 / 阶段 | 退出码 | 结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_estimators.py -q --device cpu -k 'json_key_sorting or cleared_gradient or short_last or cannot_omit' --tb=short --junitxml=artifacts/test-results/task05-review-before.xml` | 1 | 4 failed / 105 deselected；修改前复现 |
| 调度接口与 Eq.14 修改后的公式/事件 oracle 最窄组合 | 0 | 61 passed / 44 deselected |
| version 3 fixture 与核心修正后的两份算法单测 | 0 | 150 passed |
| `python -m pytest tests/unit/test_estimators.py -q --device cpu -k 'raw_byte_samples or layer_inventory_is_copied' --tb=short --junitxml=artifacts/test-results/task05-review-input-before.xml` | 1 | 4 failed / 105 deselected；修改前复现 |
| 补充输入修正后的 `python -m pytest tests/unit/test_estimators.py -q --device cpu --tb=short` | 0 | 109 passed |
| `python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/integration/test_profile_estimator.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task05-review-cpu.xml` | 1 | 161 passed / 3 errors / 0 failures / 0 skipped |
| `python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task05-review-profiler.xml` | 1 | 62 passed / 16 errors / 0 failures / 0 skipped |
| `python -m pytest tests -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task05-review-regression.xml` | 1 | 421 passed / 89 errors / 0 failures / 0 skipped；510 tests |
| `python -m compileall -q src tests`；`git diff --check` | 0 | 通过 |

- 本机仍为 Windows / Python 3.13.12 / pytest 9.1.1，无 torch；CPU 模式，没有 backend/worker/GPU/kill 启动。没有安装或改动环境、访问服务器。真实新采样逻辑、数值保持、hook 清理与 CPU profile 接入必须在有 torch 的环境执行，不能视为 CPU 验收完成。
- 三份最终 JUnit 的 errors 全部为 `No module named 'torch'`，已逐项核对；日志与 XML 同 basename，位于 `artifacts/test-results/`。新增复现记录为 `task05-review-before.log/.xml`、`task05-review-input-before.log/.xml`，汇总为 `task05-review-local-summary.json`。
- 最终审阅全部相关文件及差异，确认旧累计/平均激活路径、任意队列接口、独立梯度内存参数和旧 schema 解析均已删除。服务器从项目工作目录执行以下组合 CPU 验收；因修改 Profiler，GPU profiling 仍须按前文 Task04 合同验证。

```bash
python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/integration/test_profile_estimator.py -q --device cpu --junitxml=artifacts/test-results/task05-review-server-cpu.xml
```

## Task 06 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task 05/06、后续 Restorer/selector 接口需求及现有合同/Estimator/Profiler。适用父目录与仓库未发现额外 AGENTS.md；遵循最小实现、局部修改、现有依赖和唯一进度文件规则。保留既有 Task05 工作，本次没有修改其实现、依赖或环境。
- 算法依据核对为论文 [Algorithm 1 与 Planner 原文](https://arxiv.org/html/2508.21613v4#S4.SS1)，执行范围以总计划及 Task06 为准。没有硬编码论文或主机路径。
- 修改范围仅为新增 `src/chameleon/planner.py`、`tests/unit/test_integer_partitions.py`、`test_batch_distribution.py`、`test_layer_distribution.py`、`tests/integration/test_dynamic_planner_oracle.py`，及本进度文件。
- 整数分拆：递归枚举当前 survivor 数的非降序 pipeline lengths，全部使用现有资源，dp/pp 严格落在显式 Rdp/Rpp（支持不连续范围）；去除 pipeline 排列重复并稳定排序。每个实际 survivor 状态独立调用，不将不同故障数的候选混选为一个 runtime plan。
- batch：以整数运算按 pipeline 节点比例取 floor，递归枚举余数的全部弱组合，包括多个余数落在同一 pipeline；随后修复每个 zero partition。donor 每次重新取当前最大分区，同分取较小 pipeline ID；Nm>=dp 时守恒且全部非零，Nm<dp 返回无可行 batch。global Nm 从固定 B 与 micro-batch size 作向上整除，避免混用 sample count 和每 pipeline Nm。
- layers：blocks 先均分，枚举余数 stage 的组合，每个余数 stage 仅增加一个 block。保持完整 module_order，所有 pipeline 包含完整模型；前缀模块归首 stage，后缀（norm/head/新增末端模块）归尾 stage，层间附属模块归前一 block。允许有模块的零 block 端点 stage，排除完全空 stage。
- Planner 接入现有 Estimator Eq.10/11/14，时间和内存都包含全部端点成本；先估内存，OOM 候选保留逐 stage 原因且不伪造时间。`Planner.candidates(state)` 输出动态候选与诊断，`best_dynamic_plan(state)` 只在可行候选中按 estimated step time、稳定 plan ID 最小化；所有 OOM 或无合法分拆时抛出 `NoFeasibleDynamicPlanError`。检查 config/profile identity、layer count 和固定 B。
- plan ID 绑定完整 survivor worker identity（含 rank/generation）、布局、batch 和完整 profile 内容；输入 worker/Rdp/Rpp 排列不改变结果。输出为待 Restorer 处理的逻辑 target slots，不填写未知 transition 时间，不接收 D，不输出 score，不执行 rerouting、Equation8 policy selection 或恢复。物理 worker-to-slot 匹配、完整 survivor state sources 检查和 transition 校准继续由 Task07 实现；缓存与预计算继续由 Task15 实现。
- 独立 oracle 用笛卡尔积枚举小空间（不用生产分拆/batch/layer helper），使用手写 1F1B phase 队列与离散完成事件模拟时间，按手算公式推导内存。核对完整候选集、逐候选时间/峰值及最佳 plan；覆盖非对称 lengths、Nm<dp、零值反复修复、余数重复分配、完整端点布局、同时间 tie-break、不同 survivor count/identity、端点 OOM、全部 OOM 和容量临界点。
- 测试环境为额外开发验证：Windows / Python 3.13.12 / pytest 9.1.1，torch 不存在；device=cpu，backend/worker/GPU 均未启动。Task06 算法不依赖 torch 执行，指定 CPU 组合全部通过；这不替代 Ubuntu/Python3.12.3/pytest8.1.1 固定容器复验，也不改变 Task04/05 未验状态。没有安装、升级、降级依赖或访问服务器。

实际运行记录（每个小功能完成后执行最窄测试）：

| 命令 / 阶段 | 退出码 | 实际结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_integer_partitions.py -q --device cpu` | 0 | 136 passed |
| `python -m pytest tests/unit/test_batch_distribution.py -q --device cpu` | 0 | 73 passed |
| 初版 `python -m pytest tests/unit/test_layer_distribution.py -q --device cpu` | 1 | 45 passed / 1 failed；独立断言发现层间附属模块切分边界错误 |
| 修正边界后同一 layer 命令 | 0 | 46 passed |
| 初版 `python -m pytest tests/integration/test_dynamic_planner_oracle.py -q --device cpu --tb=short` | 1 | 24 passed / 2 failed；fixture 没有产生同时间 plan，330 字节也没有可行 plan。完整候选/时间 oracle 已通过；改用明确同时间的单 stage 双 pipeline fixture 与独立核算的 400 字节容量 |
| 修正 fixture 并加入容量临界点后的同一集成命令 | 0 | 27 passed |
| worker rank identity 补充后的最窄 `-k 'survivor_counts or exact_time_ties'` | 0 | 2 passed / 25 deselected |
| 最终 `python -m pytest tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu --junitxml=artifacts/test-results/task06-cpu.xml` | 0 | 282 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py -q --device cpu --junitxml=artifacts/test-results/task06-estimator-regression.xml` | 0 | 161 passed / 0 failed / 0 errors / 0 skipped |
| 新增文件 `python -m compileall -q ...`；AST/空白审阅；`git diff --check` | 0 | 通过 |

- 实际日志/XML 为 `artifacts/test-results/task06-cpu.log/.xml` 和 `task06-estimator-regression.log/.xml`；逐项核对 JUnit 计数与无 errors/failures/skips，汇总保存为 `task06-local-summary.json`。最终逐文件审阅新增代码与文档 diff；未引入未声明第三方库、旧接口 adapter、skip/fallback、任意 timing 常量或额外进度文档。
- 本 task 没有多进程通信、PID/端口/rendezvous/kill 资源，因此这些审计不适用；不宣称执行了真实训练/迁移或 GPU runtime 闭环。当前统一使用调用方显式提供的同质 target-slot memory capacity，Eq.14 与 computation-only 时间估计的近似边界保持不变。
- 未验证项为固定容器 CPU 复验，尚未执行服务器命令；Task04 双 GPU 校准和 Task05 真实 CPU profile 接入仍待前述验收。Task06 没有独立必测 GPU 命令，GPU runtime 闭环仍在 Task09/10；下一项为 Task07，尚未实现。

固定容器中从项目工作目录执行（使用既有 python，无需安装依赖）：

```bash
python -m pytest tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu --junitxml=artifacts/test-results/task06-server-cpu.xml
```

## Task 06 审阅与优化（2026-09-13）

- 按用户要求对照 `CLAUDE.md`、总计划和 Task06，检查整数分拆、batch 修复、完整 module layout、Estimator 组合、最优 dynamic 时间与状态绑定。修改范围仅为 `src/chameleon/planner.py`、`tests/integration/test_dynamic_planner_oracle.py` 及本进度文件；未修改既有合同、Estimator/Profiler、环境、依赖或后续任务。
- 正确性问题：原 Planner 只校验 `len(layer_modules)==num_layers`，同样数量的 embedding/head/norm 可以冒充 blocks，造成真实 block 被划入端点、层均分空间和 Eq.14 平均成本错误；倒序 blocks 也会通过入口。现按当前 TinyTransformer 的稳定模块合同，严格要求 inventory 等于配置对应的完整 `blocks.0..blocks.(L-1)` 有序序列。删除旧数量分支，不进行自动修补或兼容；新增端点替换、倒序与缺失 block 的拒绝测试。原人为修改 Estimator 内部 inventory 的测试也已替换为通过正常构造入口验证。
- 简洁性问题：原内存估计放在全 pipeline 层布局的笛卡尔积内，同一 pipeline/layout 随其他 pipeline 的组合被反复估计。现每个 topology 先生成各 pipeline 的 `(layout, memory)` 选项，再枚举组合；OOM 状态也只对组合计算一次，不在各 batch 内重复求值。没有新增持久 cache、替代算法或额外接口。三条双 stage pipeline 的独立测试仍输出全部 48 个候选，真实 Estimator 内存调用从 24 次降到 6 次。
- 完备性验证：原独立穷举/事件/内存 oracle 全部通过，继续核对完整候选集、逐候选时间与峰值、端点 OOM、容量等于上限、全部 OOM、Nm<dp、零分区修复、无合法分拆、不同 survivor 状态与稳定 tie-break。合法配置中的全部端点仍参与成本；D、transition、rerouting 与 policy selection 的边界未改变。

实际命令与结果：

| 命令 / 阶段 | 退出码 | 实际结果 |
| --- | --- | --- |
| 修改前 Task06 四文件组合（`--junitxml=artifacts/test-results/task06-review-baseline.xml`） | 0 | 282 passed |
| `python -m pytest tests/integration/test_dynamic_planner_oracle.py -q --device cpu -k 'endpoint_substitution or memory_is_not_recomputed' --tb=short --junitxml=artifacts/test-results/task06-review-before.xml` | 1 | 5 failed / 27 deselected；4 项 inventory 入口未拒绝，1 项重复内存计算（24 而非 6 次） |
| 修正后同文件 `-k 'endpoint_substitution or memory_is_not_recomputed or changed_global_batch'` | 0 | 7 passed / 26 deselected |
| `python -m pytest tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu --junitxml=artifacts/test-results/task06-review-cpu.xml` | 0 | 288 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py -q --device cpu --junitxml=artifacts/test-results/task06-review-estimator.xml` | 0 | 161 passed / 0 failed / 0 errors / 0 skipped |
| 修改 Python 文件 compileall、AST/空白审阅、`git diff --check` | 0 | 通过 |

- 已核对最终 JUnit 288/161 tests，全部无 failures/errors/skips；上述 XML 有同 basename 的真实日志，位于 `artifacts/test-results/`。汇总为 `task06-review-local-summary.json`，审阅差异为 `task06-review.diff`；修改前快照只位于忽略的 artifacts 供差异审阅，不是可执行兼容路径。
- 本机仍为 Windows / Python 3.13.12 / pytest 9.1.1、无 torch，device=cpu；没有启动 backend、worker、GPU、kill 或产生 PID/端口/rendezvous 资源，没有安装或改变环境、访问服务器。当前统一 memory capacity 和 Eq.14/computation-only 估计边界保持不变。固定容器复验及前置 Task04/05 的待验项仍未执行，不宣称真实训练/迁移验收通过；服务器沿用上节 Task06 CPU 命令执行当前实现。

## Task 07 实现与开发验证（2026-09-13）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task07、后续 selector/recovery 文档和现有 Planner/Estimator/Profiler/校准实现；适用父目录与仓库未发现额外 AGENTS.md，遵循用户提供的全局规则。保留开始时已有的 Task06 文件和进度变更，不修改后续任务或安装依赖。
- 算法依据核对为论文 [Restorer 与 Figure 3](https://arxiv.org/html/2508.21613v4#S4.SS2)。图中 DP2 到 DP2′ 的 layer-count matrix 用作独立 Hungarian 测试；实际规划按缺失 tensor 字节计费，包含 optimizer state。
- 实现/测试范围：`src/chameleon/state_sources.py`、`hungarian.py`、`coloring.py`、`restorer.py`；Task07 指定四份测试，以及补充 `tests/integration/test_restorer_inventory.py`；必要地修改 `src/chameleon/transfer_calibration.py` 和对应校准测试，更新本文件。没有修改 Planner、Estimator、模型训练路径、依赖或环境合同。
- 完整 inventory：worker 侧 `adamw_inventory` 只读当前全部 trainable parameter 和已 materialize 的 step/exp_avg/exp_avg_sq 元数据，不复制 tensor 值、不调用初始化、不读取 reference。包括 embedding、blocks、final norm、head、新增模块，排除冻结参数；缺 optimizer tensor、错误 optimizer ownership/AMSGrad/shape/step 被拒绝。
- `build_state_source_map` 对照完整必需 inventory 检查当前 survivor identity、rank/generation、committed step、tensor 名称/shape/dtype/bytes；每个模型单元必须至少有一个完整健康 parameter/AdamW 副本。任何必需 tensor 无来源即 `UnrecoverableStateError`，不拼接不同残缺副本。controller 仅保存元数据，不保存恢复用训练状态。
- Hungarian 使用整数字节与确定性下标顺序，输出 survivor-to-slot 一对一最小成本匹配。独立 permutation oracle 对照 150 个固定 seed 小矩阵，另覆盖 Figure3、零成本同分和超过浮点精度的字节成本；不引入新依赖。
- Restorer 真正接入 Planner/Estimator 输出，检查布局、survivors、generation、B、profile identity 和完整模型字节清单。manifest 每个目标 tensor 恰有一个本地保留或健康迁移来源；migration bytes 与匹配成本相等。完整旧状态列在 held_sources，release_after_ack 只列目标不再需要的旧状态，ACK 全部到齐之前禁止取得释放许可。此处是规划/许可检查，不宣称已在 worker 中实现状态保留、P2P 或 topology commit。
- 通信图覆盖 blocks 和全部端点/新增模型单元，同 worker device 的单元相邻。DSATUR 按 saturation、degree、stable ID 选点并取最小可用颜色，输出确定性 synchronization rounds；不宣称一般图最优色数。migration rounds 另按 source/destination device 冲突着色，供 transition 估计使用。
- transition 使用调用方实测且未与训练重叠的搜索时长、每轮最慢传输校准时间之和、实测 bootstrap，分别记录各成本。对应 tensor 大小缺校准就拒绝估计，包括 0 migration 时仍要求 bootstrap 校准；不使用任意带宽/时间常量或 D。当前采用同质两 rank 校准的明确 proxy：更慢方向和 endpoint mean；目标规模 bootstrap、真实物理边带宽、ACK 及 module/optimizer 容器重建额外成本仍须 runtime 实测，不宣称完整实测 transition。
- 接入时发现原 P2P 校准仅允许 FP64 整倍数字节，不能覆盖标准 AdamW 的 4 字节 step。唯一校准执行路径改为 uint8 原始字节 buffer，允许精确正整数 tensor 字节；对应非法输入改测 0 bytes，新增 4 bytes schema/lookup oracle，真实双 rank 测试改为传输 4/64/4096 bytes。保留原硬超时/finally/PID/端口/rendezvous 审计；无旧执行路径或 adapter。
- 测试环境为 Windows / Python 3.13.12 / pytest 9.1.1，无 torch，device=cpu；未安装或改变环境、访问服务器、运行 GPU/kill。Task07 指定算法组合全部通过，但本机结果不替代固定 Ubuntu/Python3.12.3/pytest8.1.1 容器复验。

实际命令与结果：

| 阶段 / 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `python -m pytest tests/unit/test_state_sources.py -q --device cpu` | 0 | 初版 42 passed |
| `python -m pytest tests/unit/test_hungarian.py -q --device cpu` | 0 | 24 passed |
| `python -m pytest tests/unit/test_coloring.py -q --device cpu` | 0 | 16 passed |
| 初版 Restorer/source 组合 | 1 | 58 passed / 2 failed；测试误用不存在的 Planner 属性，且原 fixture 没有产生可释放旧状态；修正断言和明确包含待释放状态的旧布局 |
| 修正后的 Restorer/source 组合 | 0 | 60 passed |
| 校准非 torch 最窄测试 | 0 | 39 passed / 3 deselected；仅 deselect 真实通信/清理测试，随后实际运行完整文件记录其环境错误 |
| `python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task07-cpu.xml` | 0 | 113 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task07-live-cpu.xml` | 1 | 39 passed / 20 errors / 0 failed / 0 skipped；17 项真实 inventory、3 项真实校准/清理均因缺 torch 在 setup 阶段退出 |
| `python -m pytest tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task07-regression.xml` | 1 | 511 passed / 16 errors / 0 failed / 0 skipped；16 项真实 profiling 因缺 torch 在 setup 阶段退出 |
| `python -m compileall -q src tests`；`git diff --check` | 0 | 通过 |

- 报告与完整日志：`artifacts/test-results/task07-cpu.xml/.log`、`task07-live-cpu.xml/.log`、`task07-regression.xml/.log`。JUnit 已逐项核对，错误均为 `No module named 'torch'`，没有算法 assertion failure 或 skip。
- 本次真实通信 fixture 在 import 阶段失败，没有 worker、backend、PID/端口/rendezvous 资源，不能宣称已验证字节传输或资源清理。Task07 本身没有独立 GPU 必测；实际同步与迁移继续在 Task10/12。由于修改了 Task04 校准执行路径，其双 GPU 校准同样须在既有服务器合同下复验。
- 最终检查当前相关 diff 及新增文件，保留既有工作，无额外进度文件、未声明依赖、checkpoint/reference 恢复输入、初始化恢复、mock 传输、skip/CPU fallback。下一项为 Task08 的 Equation8 selector，尚未实现。

固定容器从项目工作目录执行，使用既有 python，无需安装依赖：

```bash
python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py -q --device cpu --junitxml=artifacts/test-results/task07-server-cpu.xml
python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task07-server-live-cpu.xml
python -m pytest tests/distributed/test_transfer_calibration.py -q --device cuda --world-size 2 --require-gpu --junitxml=artifacts/test-results/task07-server-calibration-gpu.xml
```

最后一条用于受本次改动影响的 Task04 双 GPU 字节校准；不是 Task07 实际恢复/GPU runtime 已通过的证据。服务器验收未执行，完整回传日志/报告后才能确认这些待验项。

## Task 07 审阅与优化（2026-09-14）

- 按用户要求审阅正确性、完备性和简洁性，对照 `CLAUDE.md`、总计划、Task07 以及现有合同和测试。修改仅涉及 Restorer、state inventory、冲突图、必要的 Planner/Estimator 接口、校准聚合、对应测试和本文件；保留之前未提交工作，不修改依赖、环境或后续任务。
- plan 绑定问题：原 Restorer 只比较 survivor ID 和 profile identity，能接受 rank 已变化的旧 plan、identity 相同但内容变化的 profile，以及被改为 0/负数/不守恒或与时间估计不一致的 micro-batch 分配。新增 7 项测试先复现这些错误。现 DynamicPlan 保存完整 WorkerIdentity 与完整 profile hash，构造时检查 pipeline/partition geometry 和对应时间估计，Restorer 精确比较当前 survivors 与 profile 内容。
- 内部一致性问题：合法 layout 改动仍能搭配旧时间估计，可变嵌套 list 也能在生成 ID 后改变内容。现 Estimator 导出该估计对应的完整 layouts/profile hash，复制输入布局和 micro-batch 分配为 tuple；DynamicPlan 只接受不可变身份输入并核对 estimate 来源。新增测试覆盖改变合法 layout、替换 estimate profile、可变输入和输入修改后快照不漂移。plan ID 只在 DynamicPlan 构造时从当前完整内容生成，删除 Planner 的旧独立 ID 生成及裸 survivor ID 字段，不提供兼容入口。
- ACK 问题：原 ACK 只匹配 tensor/worker/slot，下一 committed step 的相同迁移会接受旧 ACK。现 manifest ID 绑定 plan、committed step 和实际 actions；ACK 必须携带该 ID，不匹配则不改变 pending 集合。更新全部调用点，删除未绑定 manifest 的 ACK 签名；保持全部 target ACK 前不能释放旧状态的规则。
- transition 问题：原 bootstrap 被标为共有成本，但混入 dynamic 的策略 transition，后续直接与 rerouting 的 paper-model 0 比较会产生不一致计费。现 `estimated_transition_time_s = unoverlapped_search_time_s + migration_time_s`，只表示 Equation8 的策略成本；`common_control_time_s` 单列实测 bootstrap，`estimated_total_time_s` 从两者求和。共有成本必须对两策略一致计入，删除旧混合 total 和 rebuild 字段；零迁移测试分别验证策略成本 0 与总成本中的实测 bootstrap。
- 实际 inventory 问题：原函数只检查 parameter step 为正，无法识别其他已提交 step 的陈旧状态；moments 也只比较 shape。现强制传入 committed step，并逐参数核对实际 AdamW step 相等及 moment shape/dtype/device 一致；移除旧“任意正 step”逻辑和多次标量读取。新增 stale positive step/dtype 的真实 PyTorch 测试及非法 committed step 的纯 CPU 合同测试。
- 冲突图问题：单个字符串 device ID 会被迭代为字符，导致不同 device 被错误连边。先运行新增测试复现，再要求 device owners 为 tuple，明确拒绝该输入；不自动猜测或修补设备信息。
- 简洁性：校准原先对每个 tensor、每个方向反复解析/校验全部样本。现唯一 `calibration_times_s` 路径一次聚合全部 endpoint means 和 bootstrap，现有 transfer/bootstrap 查询共同使用它；Restorer 在接入完整校验过的 profile 快照时保存各 size 的时间，重复估计不再解析样本。按 size 使用最新匹配的校准样本，保持较慢方向的原估计含义，不增加带宽插值或默认值。重复估计测试核对结果一致且没有重新校验 calibration。

实际命令与结果：

| 阶段 / 命令 | 退出码 | 结果 |
| --- | --- | --- |
| 修改前指定四文件组合，`--junitxml=artifacts/test-results/task07-review-baseline.xml` | 0 | 113 passed |
| `python -m pytest tests/integration/test_plan_restorer.py -q --device cpu -k 'changed_batch_distribution or binds_full_worker or binds_entire_profile or ack_is_bound or shared_bootstrap or does_not_reparse' --tb=short --junitxml=artifacts/test-results/task07-review-before.xml` | 1 | 10 failed / 26 deselected；修正前复现 |
| 修正上述问题后的同一最窄筛选 | 0 | 10 passed / 26 deselected |
| 来源/Planner/Estimator/Restorer 最窄组合 | 0 | 242 passed |
| `python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu -k 'not two_real and not hard_timeout and not worker_exception' --tb=short` | 0 | 39 passed / 3 deselected；完整文件随后在真实路径组合中实际运行 |
| 新增字符串 device ID 测试、修改输入分配后 estimate 漂移测试，分别在修正前运行 | 1 | 各 1 failed；修正前复现 |
| `python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task07-review-cpu.xml` | 0 | 135 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py -q --device cpu --tb=short --junitxml=artifacts/test-results/task07-review-algorithms.xml` | 0 | 584 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/integration/test_profile_estimator.py -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task07-review-live-regression.xml` | 1 | 101 passed / 41 errors / 0 failed / 0 skipped；19 项 inventory、3 项真实校准/清理、19 项真实 profile/Estimator 测试均因缺 torch 在 setup 阶段退出 |
| `python -m compileall -q src tests`；AST/空白审阅；`git diff --check` | 0 | 通过 |

- 已逐项核对 JUnit 135/584 tests 全部无 failures/errors/skips；真实路径的 41 errors 全部为 `No module named 'torch'`。上述 XML 具有同 basename 的完整日志；补充来源/可变输入复现报告为 `task07-review-provenance-before.xml`，汇总为 `artifacts/test-results/task07-review-local-summary.json`。
- 本机仍为 Windows / Python3.13.12 / pytest9.1.1、无 torch，device=cpu；没有 backend、worker、GPU 或 kill 启动，因此不能声称完成 PID/端口/rendezvous 清理或实际状态恢复验证。没有安装或修改环境、访问服务器。两 rank 校准的 proxy 边界仍保持，目标规模 bootstrap、真实边带宽、ACK 和 module/optimizer 容器重建耗时仍须 runtime 实测。
- 最终检查代码 diff 和新增文件，确认旧裸 worker ID 字段、未绑定 ACK 签名、旧混合 transition 字段、重复校准聚合路径均已移除，相关 producer/consumer/tests 一并更新；无 legacy adapter、初始化恢复或额外进度文件。前置 task 的服务器待验项保持原状。

固定容器从项目工作目录使用既有 python 执行当前算法与真实路径复验：

```bash
python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py -q --device cpu --junitxml=artifacts/test-results/task07-review-server-algorithms.xml
python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/integration/test_profile_estimator.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task07-review-server-live-cpu.xml
```

受 Task07 原字节校准变更影响的双 GPU 校准仍沿用上一节命令；本次未执行服务器或 GPU 验收。

## Task 08 实现与开发验证（2026-09-14）

- 已阅读 `CLAUDE.md`、`docs/MASTER_PLAN.md`、Task07/08 文档和已有 Planner、Estimator、Restorer、state sources、数据合同及测试；未发现额外仓库 AGENTS.md。保留工作区已有 Task07 未提交修改。本次仅新增 `src/chameleon/decision_center.py`、`tests/unit/test_policy_selector.py`、`tests/integration/test_decision_center_oracle.py`，更新本文件；没有修改依赖、环境或后续 task。
- `RecoveryState` 保存故障前 ClusterState、FailureEvent、原 stage 布局和完整 worker 映射，以及 survivor-only 的 parameter/AdamW 元数据。检查故障 generation、committed step、完整 worker identity、topology 覆盖和 survivor inventory；不保存训练 tensor、模型 reference 或初始化备份。评估/选择不推进 committed step、不重建 generation、不释放源状态。
- rerouting 独立保留原 layer layout 和逻辑 pipeline，健康任务仍由原 worker 执行；失败 stage 的 micro-batch 任务按稳定 worker ID 轮转给同 stage 的健康 DP peers，多故障的额外任务合并均匀分配。路线包含每个逻辑 pipeline/stage/micro-batch 的唯一 owner，forward/backward 使用同一 owner。每个 owner 必须保有该 stage 的完整 parameter/step/exp_avg/exp_avg_sq 状态。
- 使用现有 Eq.12/13，明确 global Nm 与每 pipeline Nm；使用包含 endpoints 的实测 module EMA，将最大 stage forward/backward 时间输入 uniform-stage 近似，并在候选推导中报告该边界。该 rerouting 输入表示原对称 DP/PP 布局；原 Nm 无法等分时明确标为该估计域内不可行，dynamic 仍独立评估；不为不等 partitions 伪造 Eq.12/13。Fi>=Ndp 先判不可行，避免零分母；若 survivor 额外持有旧完整单元，dynamic 仍可从 survivor memory 恢复。
- 两类候选均输出 Eq.14 内存；rerouting 静态 parameter/AdamW 只计一次，live activation 按该 worker 服务的逻辑 pipeline 数保守计入，并记录 streams、static bytes、每 stream activation bytes 和 peak。这是逻辑 tensor/平均层近似，不能声称真实 HBM 测量。无 feasible dynamic 时保留第一个被拒绝布局的数值内存诊断及 Algorithm1 全部拒绝原因，不把它称为最优 plan。
- `evaluate_candidates(state, profile)` 不接收 D、不计算 score、不选 policy。它真实调用 Algorithm1 的 `Planner.best_dynamic_plan`，仅最小化 post-recovery step time；再调用 Restorer 的 Hungarian/DSATUR manifest 和实测 transition 估计。未重叠搜索用实际 `perf_counter` 包围 Planner search；迁移仅使用精确 tensor size 的校准。缺 bootstrap/P2P 校准时 dynamic 明确不可行、transition 为 None，不填默认秒数；rerouting 的 paper-model 策略 transition 为 0，实测共有 bootstrap 对双方单列，未测时为 None。
- `select(state, profile, inter_fault_duration_s)` 先校验调用方提供的有限正 D，再严格执行 `(B/t_step)*((D-t_transition)/D)`；D<=transition 的候选没有 score。最终选最大 score，完全同分按更小 transition、再按稳定 plan ID；拒绝混合 B/generation、重复 policy/plan ID、非法或缺失估计与计算后非有限 score。缺失 D 不执行 search、不生成 DecisionResult；没有可用候选抛出带完整推导的 `NoUsablePolicyError`，没有 survivor 则抛出 `UnrecoverableStateError`。
- `PolicyDecision` 继承原 DecisionResult，携带选中的完整 rerouting routes 或 MigrationManifest，以及可 JSON 序列化的双方候选、D/B、time/transition/memory、Equation8 factors/score、唯一选中项和淘汰原因。故障 ID 集合及 survivor/inventory 顺序不改变稳定候选身份；不添加默认 policy、MTBF 预测或 min(step) 选择路径。实际 recover、P2P 和 topology commit 仍按 Task12 实现，本 task 不提供假恢复入口。
- 单测手算 B=10、rerouting(step=2,transition=0)、dynamic(step=1,transition=10)：D=15 选 rerouting、D=40 选 dynamic、D=20 同分选较小 transition。组合测试使用受控 profiling/calibration/search clock，真实执行 Planner、Estimator、Restorer；独立穷举 dynamic space、Hungarian bytes、校准 endpoint/iteration means 和窗口 useful samples oracle，不调用生产 scorer。同一 profile 下 best dynamic 与 D 无关，最终 policy 随 D 改变。覆盖单候选、两侧独立 OOM、完整副本丢失、端点及新增 trainable 单元、peer 无状态、Fi>=Ndp、无 survivor、D<=transition、缺失/非法 D、缺校准、陈旧 inventory 和 topology/profile 不匹配。

实际命令与结果：

| 阶段 / 命令 | 退出码 | 结果 |
| --- | --- | --- |
| 修改前 Task07 指定四文件 CPU 组合 | 0 | 135 passed；确认已有工作基线 |
| `python -m pytest tests/unit/test_policy_selector.py -q --device cpu` | 0 | 初版 40 passed；后补重复 plan ID 测试，最终单测 41 项包含在指定组合中 |
| 首次 `python -m pytest tests/integration/test_decision_center_oracle.py -q --device cpu --tb=short` | 1 | 35 passed / 1 failed；独立测试 oracle 漏算 calibration iteration mean 的 .5，修正 oracle 后 36 passed，生产估计未改 |
| 补充 OOM/no-peer/no-survivor 最窄筛选 | 1 / 0 | 初次 3 passed / 1 failed：fixture 故障落在较小内存 stage，无法触发预期 OOM；改为输出 stage 故障后 4 passed / 36 deselected |
| `python -m pytest tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-cpu.xml` | 0 | 81 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/unit/test_contracts.py tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-algorithm-regression.xml` | 0 | 719 passed / 0 failed / 0 errors / 0 skipped |
| `python -m compileall -q src tests`；最终源码/测试/路径审阅；`git diff --check` | 0 | 通过；未使用 Ruff 或安装第三方工具 |
| 标准库解析 JUnit 与生成本机汇总 | 1 / 0 | 初次报告字典键使用 Windows 分隔符，与相对 POSIX 键不一致而 KeyError；统一相对 `Path.as_posix()` 后核对通过并保存汇总，不影响 pytest 结果 |

- 实际环境：Windows、Python3.13.12、pytest9.1.1、device=cpu。本机 torch 不存在；此次纯算法组合不依赖 torch，没有启动 backend、worker、GPU、kill、端口或 rendezvous，因此 PID/通信清理不适用。这不替代总计划的 Python3.12.3 / pytest8.1.1 固定容器验收。
- 日志和 JUnit：`artifacts/test-results/task08-cpu.log/.xml`、`task08-algorithm-regression.log/.xml`；早期 36 项组合报告为 `task08-integration.log/.xml`。使用标准库解析最终 XML，核对 81/719 项均无 failures/errors/skips；汇总为 `artifacts/test-results/task08-local-summary.json`。
- 最终 diff 检查保留已有 Task07 修改，Task08 只涉及上述三个新增文件与本进度记录；没有绝对文件路径、未声明 import、初始化恢复、skip、mock NCCL 或环境安装操作。
- 未验证项：固定容器 Task08 复验；真实 runtime 的搜索重叠、迁移/ACK/容器重建与目标规模 group bootstrap；真实训练 adaptive switch 在 Task13 验证。本 task 不要求独立 GPU 验收，不能据此宣称后续 GPU/kill 路径通过；Task01–07 的既有服务器待验项保持原状。

固定容器从项目工作目录使用既有 python 复验：

```bash
python -m pytest tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-server-cpu.xml
```

## Task 08 审阅与优化（2026-09-14）

- 对照 `CLAUDE.md`、总计划、Task08 和已有合同，按用户要求审阅正确性、完备性及简洁性。保留已有 Task07 未提交工作；本次只修改 Decision Center、必需的 Planner/Restorer 接口、对应 selector/oracle 测试及本文件。没有改动环境、依赖、其他算法或后续任务。
- 身份与数值错配：原候选可单独改写 plan ID、B、generation、step time、memory，仍带着另一份执行明细进入选择；原 PolicyDecision 也能改写 candidate 或正 score 而通过合同。先新增测试复现，现候选绑定实际 ReroutingPlan/DynamicPlan/MigrationManifest 的身份、时间和内存；PolicyDecision 核对候选对应的 ExecutionPlan 及精确 Equation8 score。ReroutingPlan ID 在构造时由 recovery identity、profile hash 和实际 routes 生成，删除 Center 的独立有效 rerouting ID 生成路径。
- transition 来源错配：同一个 Algorithm1 plan ID 下，survivor 保有的旧 tensor 不同会产生不同 manifest 和迁移成本。新增测试证明旧代码允许混用两份 transition，导致错误 score；另测 dynamic transition 被用于 rerouting。现 PolicyCandidate 保留不可变完整 TransitionEstimate，秒数从该对象派生；Restorer 返回的 estimate 必须携带具体 manifest ID，候选核对它与执行 manifest 一致。真实 rerouting plan 的策略 transition 必须为 0、没有迁移 manifest。共有 bootstrap 与 transition 内的 common control 必须相同。
- 原状态表示不完整：旧 RecoveryState 只接受共享 stage layout，并从 global Nm 默认为各 pipeline 等分，无法表示 Planner 已支持的非对称 topology 或原不等 batch 分配。删除 `stage_modules` 输入，唯一接口改为每 pipeline 的 `layouts`、`pipeline_workers` 和显式 `pipeline_micro_batches`；校验原 geometry、完整 worker identity、每 pipeline 完整 model order、正 partitions 和 global Nm 守恒。非对称状态可继续独立搜索 dynamic；Eq.12/13 不适用于原 layouts/partitions 不同的情况时，rerouting 明确缺少适用估计并不可行，不伪造时间或重新解释数据分配。该估计域边界仍保留，未新增未经验证的非对称 rerouting 时间模型。
- OOM 与诊断路径：旧候选只把显式 reasons 视为估计缺失理由，导致已知 memory OOM 但没有时间测量的候选构造报错；决策又重复合并 memory reasons。现构造时统一合并去重 reasons，feasible 和决策淘汰原因使用同一来源。Planner 的 NoFeasibleDynamicPlanError 直接携带本轮已评估的首个 rejected plan；删除 Center 的第二次 `planner.candidates()` 枚举，保留原数值内存诊断且不将 rejected plan 当 best plan。
- 缓存与错误传播：Center 原先再次解析 calibration 计算 common bootstrap；现从 Restorer 已聚合的只读属性取值，删除 Center 的 calibration 聚合 import/调用。原广泛捕获 `ValueError` 会掩盖内部 invariant 错误并偏向 rerouting；现只有 `MissingTransitionCalibrationError` 被记录为缺数据，其余异常直接传播。TransitionEstimate 统一验证 search/migration/common 数值和总和；共有成本未测时 total 明确为 None，策略秒数仍可为 paper-model 0；删除 Restorer 重复的旧 total 局部计算/校验。
- 路由构造时一次累积每 owner 服务的逻辑 pipeline 集合，删除内存估计中逐 worker 全量重扫 routes。仍保留完整 parameter/AdamW sources、所有 endpoints/新增 trainable units、同 stage 均匀 task routing 和一次静态状态加多 stream activation 的内存估计。评估不接收 D，Algorithm1 目标不变，最终 Equation8/tie-break 不变；没有新增默认 policy、兜底恢复、初始化或兼容入口。

实际命令与结果：

| 阶段 / 命令 | 退出码 | 结果 |
| --- | --- | --- |
| 修改前指定两文件 CPU 组合，`--junitxml=artifacts/test-results/task08-review-baseline.xml` | 0 | 81 passed |
| 指定两文件最窄筛选 `-k 'decision_is_bound or memory_infeasibility_is_a_complete or candidate_metrics_and_identity or diagnostics_do_not_restart or asymmetric_original_topology' --tb=short --junitxml=artifacts/test-results/task08-review-before.xml` | 1 | 13 failed / 81 deselected；修正前复现 |
| Planner/Restorer 修改后最窄原有组合：`python -m pytest tests/integration/test_plan_restorer.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu --tb=short` | 0 | 76 passed |
| 对应 Center/输入/测试更新后的指定两文件组合 | 0 | 94 passed；上述 13 项复现均已修正 |
| manifest transition 与跨 policy transition 的独立复现，各自单项筛选 | 1 / 0 | 两项各在修正前 1 failed，修正后各 1 passed；分别保存 before XML |
| `python -m pytest tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-review-cpu.xml` | 0 | 最终 108 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/unit/test_contracts.py tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py tests/unit/test_integer_partitions.py tests/unit/test_batch_distribution.py tests/unit/test_layer_distribution.py tests/integration/test_dynamic_planner_oracle.py tests/unit/test_1f1b_schedule.py tests/unit/test_estimators.py tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-review-algorithms.xml` | 0 | 最终 746 passed / 0 failed / 0 errors / 0 skipped |
| `python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --tb=short --junitxml=artifacts/test-results/task08-review-live-cpu.xml` | 1 | 39 passed / 22 errors / 0 failed / 0 skipped；19 项真实 inventory、3 项真实通信/清理均因缺 torch 在 setup 阶段退出 |
| `python -m compileall -q src tests`；源码/调用点/空白审阅；`git diff --check`；标准库解析 JUnit | 0 | 通过；108/746 报告均无 failures/errors/skips；22 项 setup errors 均为 No module named 'torch' |

- 日志/报告：`artifacts/test-results/task08-review-cpu.log/.xml`、`task08-review-algorithms.log/.xml`、`task08-review-live-cpu.log/.xml`；复现 XML 为 `task08-review-before.xml`、`task08-review-transition-before.xml`、`task08-review-rerouting-transition-before.xml`。环境及报告汇总为 `artifacts/test-results/task08-review-local-summary.json`。
- 本机仍为 Windows / Python3.13.12 / pytest9.1.1、缺少 torch，device=cpu。没有创建 backend、worker、GPU、kill、端口或 rendezvous，实际 PID/通信清理尚未验证；不以受控 profiling/timing metadata 或本机通过代替固定容器或真实训练恢复。没有安装/修改环境或访问服务器。
- 最终审阅确认旧 RecoveryState 共享布局签名、独立 transition 标量字段、重复 reason 合并、二次 Planner search、重复 common calibration 聚合、捕获全部 ValueError 的降级分支及旧 total 计算已移除；全部现有 producer/consumer/tests 使用新接口，没有 legacy adapter。Task01–07 既有服务器待验项不变，真实 adaptive switch 仍在 Task13。

固定容器从项目工作目录使用既有 python 复验：

```bash
python -m pytest tests/unit/test_policy_selector.py tests/integration/test_decision_center_oracle.py tests/integration/test_plan_restorer.py tests/integration/test_dynamic_planner_oracle.py -q --device cpu --junitxml=artifacts/test-results/task08-review-server-cpu.xml
python -m pytest tests/integration/test_restorer_inventory.py tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2 --junitxml=artifacts/test-results/task08-review-server-live-cpu.xml
```

## 每次完成小功能的记录格式

- Task / 小功能：
- 修改范围：
- 测试环境（Python/torch/device/backend/GPU 数）：
- 实际测试命令：
- 退出码及 passed/failed/skipped 数：
- 组合回归命令与结果：
- 报告/日志位置：
- kill PID/exitcode、资源清理结果（如适用）：
- 未验证项与失败原因：
- 下一项依赖：
