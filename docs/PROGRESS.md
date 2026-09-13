# 项目进度

本文件是唯一进度记录位置。Task 01 的工程骨架、数据合同和环境测试已实现；Task 02 的模型、确定性数据、单进程 reference 和 step commit 已实现；Task 03 的全局 loss/sample accounting、单设备 owner gradient SUM 与一次归一化及对照测试已实现。本机无需 torch 的单测已通过，训练数值测试因缺少 torch 尚未执行成功；真实 CPU/Gloo、服务器 GPU/NCCL 和恢复验收仍待执行。按用户要求，GPU 测试由用户在服务器启动，本机不继续执行 GPU 测试。未实现分布式训练或恢复算法，未执行真实 GPU 或训练进程 kill 测试。

## 状态

| Task | 状态 | 实际测试 |
| --- | --- | --- |
| 01 | 已修正 NCCL 元数据判断；非 torch 合同已验；真实 CPU/GPU 待验 | 最新全量回归中 84 passed / 13 errors；9 项新增元数据测试及 4 项真实 smoke 因缺少 torch 报错；GPU 未执行 |
| 02 | 已修正 optimizer 完成与 commit 的时序；非 torch 单测已验；CPU/GPU 数值待验 | 最新全量回归中 17 passed / 15 errors；训练/模型测试因缺少 torch 报错；GPU 未执行 |
| 03 | 已替换 owner 清单接口和存储重叠检查；ID/清单单测已验；CPU/GPU 数值待验 | 最新 Task 03 组合 32 passed / 39 errors；完整回归 133 passed / 67 errors；错误全部为缺少 torch；GPU 未执行 |
| 04-15 | 待实施 | 未执行 |

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
