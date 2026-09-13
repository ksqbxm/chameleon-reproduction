# Task 02：确定性模型、数据与单进程 reference

前置：01。约束：[总计划](../MASTER_PLAN.md)。

## 功能步骤

1. 实现 Embedding → Transformer blocks → final LayerNorm → LM head，初始版本不绑定 embedding/head 权重；全部 dropout=0。
2. 为所有 trainable parameters 定义稳定名称/模块身份，建立完整 inventory，不只统计 blocks。
3. 固定 B 和序列长度；sample ID 仅由 committed global step 决定，数据由 ID 的确定性函数产生，不依赖数据 RNG/cursor。
4. 实现单进程 AdamW reference，保留 sample loss sum、完整参数/梯度/优化器 state 供测试读取。
5. 实现 step commit：全部参与者确认后才推进 committed step；初始化 seed 仅用于第一次建模。

每个步骤完成立即运行对应测试；reference 不得依赖待测分布式实现。

## 必须运行

```powershell
python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cpu
```

服务器：

```bash
python -m pytest tests/unit/test_model_data.py tests/unit/test_reference.py -q --device cuda --world-size 1 --require-gpu
```

## 测试场景与完成标准

- inventory 覆盖 requires_grad parameters 的精确集合，含 embedding、final norm、head。
- 检查所有 dropout 模块 p=0；训练 forward 无随机结果漂移。
- next IDs 精确等于 [step*B,(step+1)*B)，重建/失败不提前推进；连续 step 不重复/漏样本。
- 完成至少 3 个 step 后 AdamW step、exp_avg、exp_avg_sq 均非默认初始化值。
- CPU FP64/GPU reference 可重复；优化器提交前后的数据位置明确。
