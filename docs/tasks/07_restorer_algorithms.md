# Task 07：迁移匹配、state sources 与通信着色

前置：06。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 从完整 trainable/AdamW inventory 建立 survivor StateSourceMap；缺少任何所需 tensor 在规划阶段即失败。
2. 构造 survivor-to-target-slot cost matrix，已拥有状态无传输成本；缺少 parameter + optimizer tensors 按 bytes 计。
3. 实现确定性 Hungarian/Kuhn-Munkres；输出 retention/migration manifest 和每 tensor source。
4. 构建 layer/model-unit 冲突图，同 device 参数单元之间连边；端点同步亦需覆盖。
5. DSATUR 贪心按 saturation/degree/stable ID 选点，输出 rounds；不宣称一般图最小着色。
6. 用 profiling transfer/bootstrap time 估计 dynamic transition；缺校准拒绝估计，不能填魔法常数。

## 必须运行

```powershell
python -m pytest tests/unit/test_state_sources.py tests/unit/test_hungarian.py tests/unit/test_coloring.py tests/integration/test_plan_restorer.py -q --device cpu
```

## 测试场景与完成标准

- 论文 Figure 3 layer-count matrix；另测实际 bytes、optimizer states 改变迁移成本。
- 小矩阵 permutation 穷举 oracle 验证最小总成本，随机小矩阵固定 seed。
- 全部 trainable，包括 embedding/norm/head 缺 source 分别触发 UnrecoverableStateError。
- manifest 无多余/遗漏，保留源至 ACK，不删除迁移前需要的 layer。
- 路径/环/完全图和非对称 partitions：相邻异色、同 round 无 device 冲突、结果稳定。
- 实际 AllReduce 和 P2P 分别在 10/12 强制运行；本 task 只是算法门槛。
