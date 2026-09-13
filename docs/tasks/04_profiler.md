# Task 04：Profiler

前置：03。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 收集每层/端点模块 forward/backward execution time；CPU monotonic clock，CUDA events 同步后读取。
2. 收集 parameter/gradient/AdamW/activation tensor bytes；AdamW 必须预热 materialize 后再统计。
3. CUDA 采集 peak allocated/reserved HBM，CPU 标注 logical tensor bytes，禁止假装 CPU HBM 实测。
4. 收集 parallel configuration、step timing、真实 1F1B operation trace；运行时内存聚合原始样本与 EMA。
5. 校准 P2P execution time 和 group bootstrap timing，供 Restorer 秒级估计，不能随意填带宽。
6. 显式导出/加载 versioned JSON，验证 schema/model/config/device identity；正常训练不每步刷盘。

## 必须运行

```powershell
python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py -q --device cpu
python -m pytest tests/distributed/test_transfer_calibration.py -q --device cpu --world-size 2
```

服务器：

```bash
python -m pytest tests/unit/test_profiler.py tests/integration/test_profile_roundtrip.py tests/distributed/test_transfer_calibration.py -q --device cuda --world-size 2 --require-gpu
```

## 测试场景与完成标准

- 参数/梯度/optimizer bytes 与独立 tensor inventory 一致，包含端点模块。
- timing 有限非负、EMA 可手算、CUDA events 实际记录、memory peak 合理且无长期泄漏。
- JSON round-trip、陈旧模型/hash、设备不同、字段/版本错误覆盖。
- 两个真实 rank 传输 tensor 并记录 timing，不 mock P2P。
- 不采集或预测 MTBF/D；profile 文件不是恢复 checkpoint。
