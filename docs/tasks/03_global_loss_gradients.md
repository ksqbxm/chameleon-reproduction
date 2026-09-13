# Task 03：global loss sum 与梯度归一化

前置：02。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 每个 sample 内对有效 token loss 求平均；micro-batch 只对 sample scalars 求和，不作 sample mean。
2. 实现 sample-count accounting，验证全部 pipeline/rerouted IDs 是当前 global IDs 的不重不漏分区。
3. 实现梯度 sum accumulation，所有 owner 全局 SUM 后只除一次 global sample count。
4. 同一 accounting 同时供 dynamic 和 rerouting 使用；计数由实际 IDs/大小验证，不信任局部配置猜测。
5. 汇总报告 loss_global_sum、global_sample_count、loss_global_mean，禁止平均 pipeline means。

## 必须运行

```powershell
python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cpu
```

服务器：

```bash
python -m pytest tests/unit/test_global_loss.py tests/integration/test_partitioned_gradient_oracle.py -q --device cuda --world-size 1 --require-gpu
```

## 测试场景与完成标准

- 明确构造 [5,3,2] 的 micro-batch 分区，结果等于 global sum/10；错误“均值的均值”结果必须与正确结果不同。
- 不等 micro-batch sample sizes、最后 partial micro-batch 按 samples 权重正确。
- 丢失/重复 sample ID、重复归一化、计数不一致明确失败。
- 至少 3 步参数和 AdamW state 对照未分区 reference。
- 本 task 为单 device 数学门槛；实际 distributed dynamic/rerouting 必须在 10/11 再证明，不能用本测试替代。
