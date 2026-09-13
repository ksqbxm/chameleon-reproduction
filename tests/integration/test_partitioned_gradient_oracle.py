"""Single-device mathematical replicas; no worker or distributed-runtime claim."""

from copy import deepcopy
import math

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.data import make_batch, next_sample_ids
from chameleon.global_loss import GlobalBatchAccounting
from chameleon.step import StepCommit


@pytest.fixture
def torch_module():
    import torch
    return torch


@pytest.mark.parametrize("policy", ["dynamic", "rerouting"])
@pytest.mark.parametrize("sizes", [
    ((1, 1, 1, 1, 1), (1, 1, 1), (1, 1)),
    ((2, 1, 3, 2, 1), (4, 1, 2), (3, 2)),
], ids=["five-three-two", "unequal-partial-samples"])
def test_three_steps_match_unpartitioned_reference(torch_module, device, policy, sizes):
    from chameleon.global_loss import micro_batch_loss_sum
    from chameleon.model import build_initial_model, parameter_inventory
    from chameleon.reference import ReferenceTrainer

    torch = torch_module
    global_count = sum(sum(pipeline) for pipeline in sizes)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=1, num_heads=1,
                         sequence_length=3, global_batch_size=global_count, micro_batch_size=4)
    state = ClusterState((WorkerIdentity("single-device", 0, 0),), global_count)
    reference = ReferenceTrainer(build_initial_model(config, device=device), state,
                                 lr=0.007, weight_decay=0.06)
    reference.optimizer.param_groups[0].update(betas=(0.8, 0.95), eps=1e-7)
    # Rerouting's peer accumulates two logical pipelines into the SAME buffer.
    # Full-model replicas here check algebra only; stage-level P2P is Task 11.
    pipeline_to_owner = (0, 1, 2) if policy == "dynamic" else (0, 1, 0)
    models = [deepcopy(reference.model) for _ in range(max(pipeline_to_owner) + 1)]
    owner_ids = tuple(f"owner{i}" for i in range(len(models)))
    expected_owners = {owner_id: {entry.name for entry in parameter_inventory(model)}
                       for owner_id, model in zip(owner_ids, models)}
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.007, weight_decay=0.06,
                                   betas=(0.8, 0.95), eps=1e-7, amsgrad=False) for model in models]
    commit = StepCommit(state)
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    names = {entry.name for entry in parameter_inventory(reference.model)}
    assert {name.split(".")[0] for name in names} == {
        "embedding", "blocks", "final_norm", "lm_head",
    }
    all_ids = []
    for step in range(1, 4):
        ids = next_sample_ids(commit.state)
        assert ids == next_sample_ids(reference.commit.state)
        ledger = GlobalBatchAccounting(commit.state, expected_owners=expected_owners)
        for model, optimizer in zip(models, optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
        offset = 0
        pipeline_sums = []
        pipeline_gradients = []
        actual_partition = []
        for pipeline, micro_sizes in enumerate(sizes):
            model = models[pipeline_to_owner[pipeline]]
            before = {name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                      for name, p in model.named_parameters()}
            micro_sums = []
            partition = []
            for size in micro_sizes:
                micro_ids = ids[offset:offset + size]
                offset += size
                batch = make_batch(micro_ids, config, device=device)
                loss_sum = micro_batch_loss_sum(model(batch.inputs), batch.targets)
                ledger.add_micro_batch(batch.sample_ids, loss_sum.item(),
                                       sample_count=batch.inputs.shape[0])
                loss_sum.backward()
                micro_sums.append(loss_sum.item())
                partition.append(micro_ids)
            actual_partition.append(tuple(partition))
            pipeline_sums.append(math.fsum(micro_sums))
            pipeline_gradients.append({name: p.grad.detach().clone() - before[name]
                                       for name, p in model.named_parameters()})
        assert [len(pipeline) for pipeline in actual_partition] == [5, 3, 2]
        assert tuple(i for pipeline in actual_partition for micro in pipeline for i in micro) == ids
        assert offset == global_count
        buffers = {owner_id: {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
                   for owner_id, model in zip(owner_ids, models)}
        report = ledger.sum_and_normalize_gradients_(buffers)
        assert report.global_sample_count == global_count
        assert report.loss_global_sum == pytest.approx(math.fsum(pipeline_sums),
                                                      rel=tolerance["rtol"], abs=tolerance["atol"])
        # Independent negative control: averaging pipeline means changes both
        # the reported loss and the gradient when pipelines have unequal sizes.
        wrong_loss = sum(loss / sum(partition) for loss, partition in zip(pipeline_sums, sizes)) / 3
        wrong_gradients = {
            name: sum(gradient[name] / sum(partition)
                      for gradient, partition in zip(pipeline_gradients, sizes)) / 3
            for name in names
        }
        assert wrong_loss != pytest.approx(report.loss_global_mean, rel=tolerance["rtol"],
                                          abs=tolerance["atol"])
        assert any(not torch.allclose(buffers[owner_ids[0]][name], wrong_gradients[name], **tolerance)
                   for name in names)
        for optimizer in optimizers:
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize(next(models[0].parameters()).device)
        assert commit.state.committed_global_step == step - 1
        assert next_sample_ids(commit.state) == ids
        assert commit.acknowledge(state.workers[0], step)
        expected = reference.train_step()
        assert expected.sample_ids == ids
        assert expected.committed_global_step == commit.state.committed_global_step == step
        assert expected.global_sample_count == report.global_sample_count
        assert report.loss_global_sum == pytest.approx(expected.loss_global_sum,
                                                      rel=tolerance["rtol"], abs=tolerance["atol"])
        assert report.loss_global_mean == pytest.approx(expected.loss_global_mean,
                                                       rel=tolerance["rtol"], abs=tolerance["atol"])
        assert set(expected.parameters) == set(expected.gradients) == set(expected.optimizer_state) == names
        for model, optimizer, gradients in zip(models, optimizers, buffers.values()):
            parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
            assert set(parameters) == set(gradients) == names
            assert set(optimizer.state) == set(parameters.values())
            for name, parameter in parameters.items():
                torch.testing.assert_close(gradients[name].detach().cpu(), expected.gradients[name],
                                           **tolerance)
                torch.testing.assert_close(parameter.detach().cpu(), expected.parameters[name], **tolerance)
                actual_state = optimizer.state[parameter]
                assert set(actual_state) == set(expected.optimizer_state[name]) == {
                    "step", "exp_avg", "exp_avg_sq",
                }
                assert actual_state["step"].item() == step
                for key, value in actual_state.items():
                    torch.testing.assert_close(value.detach().cpu(), expected.optimizer_state[name][key],
                                               **tolerance)
        all_ids.extend(ids)
    assert all_ids == list(range(3 * global_count))
    assert next_sample_ids(commit.state) == tuple(range(3 * global_count, 4 * global_count))
