from copy import deepcopy
from dataclasses import replace
import math

import pytest

from chameleon import ClusterState, FailureEvent, ModelConfig, WorkerIdentity
from chameleon.data import next_sample_ids
from chameleon.step import StepCommit


def cluster(size=3, *, step=0):
    return ClusterState(tuple(WorkerIdentity(f"w{i}", i, 0) for i in range(size)), 5,
                        committed_global_step=step)


@pytest.fixture
def torch_module():
    import torch
    return torch


@pytest.fixture
def reference(torch_module, device):
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=1, num_heads=1,
                         sequence_length=3, global_batch_size=5, micro_batch_size=2)
    return ReferenceTrainer(build_initial_model(config, device=device), cluster(1))


def test_commit_requires_every_participant():
    state = cluster()
    commit = StepCommit(state)
    for completed_step in range(1, 4):
        expected = tuple(range((completed_step - 1) * 5, completed_step * 5))
        assert next_sample_ids(commit.state) == expected
        assert not commit.acknowledge(state.workers[2], completed_step)
        assert not commit.acknowledge(state.workers[0], completed_step)
        assert commit.state.committed_global_step == completed_step - 1
        assert next_sample_ids(commit.state) == expected
        assert commit.acknowledge(state.workers[1], completed_step)
        assert commit.state.committed_global_step == completed_step
        assert next_sample_ids(commit.state) == tuple(range(completed_step * 5, (completed_step + 1) * 5))
    assert state.committed_global_step == 0


def test_duplicate_confirmation_cannot_replace_a_missing_worker():
    state = cluster()
    commit = StepCommit(state)
    assert not commit.acknowledge(state.workers[0], 1)
    with pytest.raises(ValueError, match="already acknowledged"):
        commit.acknowledge(state.workers[0], 1)
    assert commit.state == state
    assert not commit.acknowledge(state.workers[1], 1)
    assert commit.acknowledge(state.workers[2], 1)


@pytest.mark.parametrize("worker", [WorkerIdentity("unknown", 0, 0),
                                 WorkerIdentity("w0", 0, 1), WorkerIdentity("w0", 1, 0)])
def test_confirmation_checks_persistent_identity_and_generation(worker):
    state = cluster()
    commit = StepCommit(state)
    with pytest.raises(ValueError, match="identity"):
        commit.acknowledge(worker, 1)
    assert commit.state == state


@pytest.mark.parametrize("step", [0, -1, True, 1.5, 2])
def test_confirmation_rejects_invalid_or_future_steps(step):
    state = cluster()
    commit = StepCommit(state)
    with pytest.raises(ValueError, match="step"):
        commit.acknowledge(state.workers[0], step)
    assert commit.state == state


def test_old_confirmation_is_rejected_after_commit():
    state = cluster(1)
    commit = StepCommit(state)
    assert commit.acknowledge(state.workers[0], 1)
    with pytest.raises(ValueError, match="next global step"):
        commit.acknowledge(state.workers[0], 1)
    assert commit.state.committed_global_step == 1


def test_failure_and_topology_rebuild_do_not_advance_data():
    state = cluster(step=3)
    commit = StepCommit(state)
    failure = FailureEvent(("w1",), state.generation, state.committed_global_step)
    assert commit.state == state
    assert failure.committed_global_step == 3
    rebuilt_state = replace(state, generation=1, workers=tuple(
        replace(w, generation=1, rank=i) for i, w in enumerate(state.workers)
        if w.worker_id not in failure.failed_worker_ids
    ))
    rebuilt = StepCommit(rebuilt_state)
    assert next_sample_ids(rebuilt.state) == next_sample_ids(commit.state) == (15, 16, 17, 18, 19)
    with pytest.raises(ValueError, match="identity"):
        rebuilt.acknowledge(state.workers[0], 4)
    assert not rebuilt.acknowledge(rebuilt.state.workers[0], 4)
    assert rebuilt.state.committed_global_step == 3


def test_reference_loss_is_a_sum_of_sample_token_means(torch_module, device):
    from chameleon.reference import sample_losses
    torch = torch_module
    logits = torch.tensor([[[0., 0.], [0., math.log(3)], [0., 0.]],
                           [[0., math.log(3)], [0., 0.], [0., 0.]]],
                          dtype=torch.float64, device=device, requires_grad=True)
    targets = torch.tensor([[0, 1, -100], [0, -100, -100]], device=device)
    losses = sample_losses(logits, targets)
    expected = torch.tensor([(math.log(2) + math.log(4 / 3)) / 2, math.log(4)],
                            dtype=torch.float64, device=device)
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(losses, expected, **tolerance)
    assert losses.sum().item() == pytest.approx(expected.sum().item(), rel=tolerance["rtol"])
    losses.sum().backward()
    assert torch.count_nonzero(logits.grad[targets == -100]).item() == 0
    with pytest.raises(ValueError, match="valid token"):
        sample_losses(logits, torch.full_like(targets, -100))


def test_three_steps_populate_all_adamw_states(reference, torch_module):
    from chameleon.model import parameter_inventory
    torch = torch_module
    names = {p.name for p in parameter_inventory(reference.model)}
    initial = {name: p.detach().cpu().clone() for name, p in reference.model.named_parameters()}
    assert not reference.optimizer.state
    all_ids = []
    first = None
    for step in range(1, 4):
        assert next_sample_ids(reference.commit.state) == tuple(range((step - 1) * 5, step * 5))
        result = reference.train_step()
        assert result.committed_global_step == reference.commit.state.committed_global_step == step
        assert result.sample_ids == tuple(range((step - 1) * 5, step * 5))
        all_ids.extend(result.sample_ids)
        assert result.global_sample_count == 5
        assert result.loss_global_sum == pytest.approx(result.sample_losses.sum().item())
        assert result.loss_global_mean == pytest.approx(result.loss_global_sum / 5)
        assert set(result.parameters) == set(result.gradients) == set(result.optimizer_state) == names
        for name in names:
            state = result.optimizer_state[name]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
            assert state["step"].item() == step
            assert torch.count_nonzero(state["exp_avg"]).item() > 0
            assert torch.count_nonzero(state["exp_avg_sq"]).item() > 0
            assert torch.isfinite(result.parameters[name]).all()
            assert torch.isfinite(result.gradients[name]).all()
        if first is None:
            first = result
    assert all_ids == list(range(15))
    assert next_sample_ids(reference.commit.state) == (15, 16, 17, 18, 19)
    assert all(not torch.equal(initial[name], result.parameters[name]) for name in names)
    assert all(state["step"].item() == 1 for state in first.optimizer_state.values())


def test_reference_is_reproducible_without_reseeding_each_step(reference, torch_module, device, monkeypatch):
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    torch = torch_module
    second = ReferenceTrainer(build_initial_model(reference.model.config, device=device), cluster(1))

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not initialize or reseed the model")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr("chameleon.model.build_initial_model", forbidden)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state() if device == "cuda" else None
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    for _ in range(3):
        left, right = reference.train_step(), second.train_step()
        assert left.sample_ids == right.sample_ids
        assert left.loss_global_sum == pytest.approx(right.loss_global_sum, rel=tolerance["rtol"],
                                                   abs=tolerance["atol"])
        for name in left.parameters:
            torch.testing.assert_close(left.parameters[name], right.parameters[name], **tolerance)
            torch.testing.assert_close(left.gradients[name], right.gradients[name], **tolerance)
            for key in left.optimizer_state[name]:
                torch.testing.assert_close(left.optimizer_state[name][key], right.optimizer_state[name][key],
                                           **tolerance)
    assert torch.equal(before, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())


def test_reference_matches_independent_loss_gradients_and_adamw(reference, torch_module, device):
    from chameleon.data import make_batch
    torch = torch_module
    oracle = deepcopy(reference.model)
    moments = {name: (torch.zeros_like(p), torch.zeros_like(p)) for name, p in oracle.named_parameters()}
    group = reference.optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    for step in range(1, 4):
        batch = make_batch(range((step - 1) * 5, step * 5), oracle.config, device=device)
        oracle.zero_grad(set_to_none=True)
        logits = oracle(batch.inputs)
        # All generated tokens are valid. This independent expression averages
        # the token loss within each sequence, then averages the five samples.
        nll = -logits.log_softmax(dim=-1).gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
        expected_loss = nll.mean(dim=1).mean()
        expected_loss.backward()
        expected_gradients = {name: p.grad.detach().clone() for name, p in oracle.named_parameters()}
        with torch.no_grad():
            for name, parameter in oracle.named_parameters():
                first, second = moments[name]
                gradient = expected_gradients[name]
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                denominator = (second / (1 - beta2 ** step)).sqrt() + group["eps"]
                parameter.addcdiv_(first / (1 - beta1 ** step), denominator, value=-group["lr"])
        actual = reference.train_step()
        assert actual.loss_global_mean == pytest.approx(expected_loss.item(), rel=tolerance["rtol"],
                                                      abs=tolerance["atol"])
        for name, parameter in oracle.named_parameters():
            torch.testing.assert_close(actual.gradients[name], expected_gradients[name].cpu(), **tolerance)
            torch.testing.assert_close(actual.parameters[name], parameter.detach().cpu(), **tolerance)
            torch.testing.assert_close(actual.optimizer_state[name]["exp_avg"], moments[name][0].cpu(),
                                       **tolerance)
            torch.testing.assert_close(actual.optimizer_state[name]["exp_avg_sq"], moments[name][1].cpu(),
                                       **tolerance)


def test_optimizer_failure_does_not_commit_or_consume_ids(reference, monkeypatch):
    ids = next_sample_ids(reference.commit.state)
    original_step = reference.optimizer.step

    def fail_before_update():
        raise RuntimeError("injected pre-update failure")

    monkeypatch.setattr(reference.optimizer, "step", fail_before_update)
    with pytest.raises(RuntimeError, match="pre-update failure"):
        reference.train_step()
    assert reference.commit.state.committed_global_step == 0
    assert next_sample_ids(reference.commit.state) == ids
    assert not reference.optimizer.state

    def observe_update_before_commit():
        assert next_sample_ids(reference.commit.state) == ids
        original_step()
        assert reference.commit.state.committed_global_step == 0
        assert next_sample_ids(reference.commit.state) == ids

    monkeypatch.setattr(reference.optimizer, "step", observe_update_before_commit)
    assert reference.train_step().sample_ids == ids
    assert reference.commit.state.committed_global_step == 1
    assert next_sample_ids(reference.commit.state) == (5, 6, 7, 8, 9)


def test_reference_rejects_mismatched_batch_and_multiple_participants(reference):
    from chameleon.reference import ReferenceTrainer
    with pytest.raises(ValueError, match="one participant"):
        ReferenceTrainer(reference.model, cluster(2))
    with pytest.raises(ValueError, match="global batch size"):
        ReferenceTrainer(reference.model, replace(cluster(1), global_batch_size=6))
