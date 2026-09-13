# Task 11：真实stage-level data rerouting

前置：10。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 逻辑pipeline/stage与物理rank分离，为缺失逻辑stage建立健康同stage peer routing。
2. 原pipeline的健康前/后stage继续执行；仅缺失stage任务转给peer，不能偷偷丢弃整个pipeline或改成纯DP。
3. peer真实接收额外activation、执行forward/backward并回传对应逻辑pipeline。
4. 原生与rerouted任务共享本地参数并累积gradient SUM；相同参数健康owners全局SUM，除global sample count一次。
5. 检查route无重复任务/错micro-batch ID，Fi>=Ndp报不可行，不重分层冒充rerouting。

## 必须运行

本task直接启动一个缺失逻辑stage的配置，验证真实通信和计算；不把它当真实kill验收。

```powershell
python -m pytest tests/distributed/test_rerouted_training.py tests/integration/test_routing_accounting.py -q --device cpu --world-size 5
```

服务器：

```bash
python -m pytest tests/distributed/test_rerouted_training.py tests/integration/test_routing_accounting.py -q --device cuda --world-size 5 --require-gpu
python -m pytest tests/distributed/test_rerouting_scale.py -q --device cuda --world-size 7 --require-gpu
```

## 测试场景与完成标准

- 5个真实workers服务DP3/PP2的6个逻辑slots，1个缺失stage由peer承担；batch [5,3,2]。
- 7个真实GPU workers服务DP4/PP2的8个逻辑slots，验证更多peers均匀分担。
- 不使用idle备份worker填补缺失slot；initial topology明确记录逻辑与物理数量。
- 不等micro-batch size、额外任务、多个不同stage缺失的可恢复routing。
- 真实参数/gradients/loss/AdamW至少3 steps对照reference，所有trainable端点包含在内。
- layout保持不变、sample IDs守恒；真实kill与survivor state来源将在12/13验证。
