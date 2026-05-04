import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "wandb",
    types.SimpleNamespace(run=None, log=lambda *args, **kwargs: None),
)

torch = pytest.importorskip("torch")

from k_utils import resolve_k_value
from optimizers.jaguar_signsgd import Jaguar_SignSGD
from optimizers.sparse_jaguar_signsgd import Sparse_Jaguar_SignSGD
from optimizers.zo_adamm import ZO_AdaMM
from optimizers.zo_rl_jaguar import ZO_RL_Jaguar
from optimizers.zo_rl_adamm import ZO_RL_AdaMM
from optimizers.zo_rl_sgd import ZO_RL_SGD
from optimizers.zo_sgd import ZO_SGD


def _patch_randint(monkeypatch, seeds):
    seed_iter = iter(seeds)

    def fake_randint(*args, **kwargs):
        return next(seed_iter)

    monkeypatch.setattr(np.random, "randint", fake_randint)


def _make_quadratic_closure(param):
    calls = {"count": 0}

    def closure():
        calls["count"] += 1
        return param.square().sum()

    return closure, calls


def _sample_dense_direction(shape, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.normal(mean=0, std=1, size=shape, generator=generator)


def _dense_probe_average(theta, eps, seeds):
    losses = []
    grads = []
    for seed in seeds:
        z = _sample_dense_direction(theta.shape, seed)
        loss_plus = (theta + eps * z).square().sum()
        loss_minus = (theta - eps * z).square().sum()
        projected_grad = (loss_plus - loss_minus) / 2

        losses.append(loss_plus)
        grads.append(z * projected_grad / eps)

    return torch.stack(losses).mean(), torch.stack(grads).mean(dim=0)


def _dense_probe_single(theta, eps, seed):
    z = _sample_dense_direction(theta.shape, seed)
    loss_plus = (theta + eps * z).square().sum()
    loss_minus = (theta - eps * z).square().sum()
    projected_grad = (loss_plus - loss_minus) / 2
    grad = z * projected_grad / eps
    return loss_plus, grad


def _sparse_indices(length, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator)[: max(1, int(length * 0.1))]


def _sparse_probe_average(theta, eps, seeds):
    losses = []
    grads = []
    touched = torch.zeros_like(theta, dtype=torch.bool)

    for seed in seeds:
        indices = _sparse_indices(theta.numel(), seed)
        loss_plus = theta.clone()
        loss_plus[indices] += eps
        loss_plus = loss_plus.square().sum()

        loss_minus = theta.clone()
        loss_minus[indices] -= eps
        loss_minus = loss_minus.square().sum()

        grad_sparse = torch.zeros_like(theta)
        grad_sparse[indices] += ((loss_plus - loss_minus) / 2).item() / eps

        losses.append(loss_plus)
        grads.append(grad_sparse)
        touched[indices] = True

    return torch.stack(losses).mean(), torch.stack(grads).mean(dim=0), touched


@pytest.mark.parametrize("k,seeds", [(1, [123]), (2, [123, 456])])
def test_zo_sgd_matches_manual_probe_average(monkeypatch, k, seeds):
    _patch_randint(monkeypatch, seeds)

    theta0 = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float32)
    param = torch.nn.Parameter(theta0.clone())
    optimizer = ZO_SGD([param], lr=0.1, eps=1e-3, momentum=0.0, k=k)
    closure, calls = _make_quadratic_closure(param)

    returned_loss = optimizer.step(closure)

    expected_loss, expected_grad = _dense_probe_average(theta0, 1e-3, seeds)
    expected_param = theta0 - 0.1 * expected_grad

    assert calls["count"] == 2 * k
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_zo_sgd_replays_distinct_param_vectors_for_update(monkeypatch):
    _patch_randint(monkeypatch, [123])

    param_a = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.5, -0.1], dtype=torch.float32))
    optimizer = ZO_SGD([param_a, param_b], lr=0.1, eps=1e-3, momentum=0.0, k=1)

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    assert len(sample_calls) == 8
    assert not torch.allclose(sample_calls[0], sample_calls[1])
    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 4])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 6])


@pytest.mark.parametrize("k,seeds", [(1, [321]), (2, [321, 654])])
def test_zo_adamm_matches_manual_probe_average(monkeypatch, k, seeds):
    _patch_randint(monkeypatch, seeds)

    theta0 = torch.tensor([0.4, -0.1, 0.2], dtype=torch.float32)
    param = torch.nn.Parameter(theta0.clone())
    optimizer = ZO_AdaMM([param], lr=0.05, eps=1e-3, betas=(0.9, 0.999), k=k)
    closure, calls = _make_quadratic_closure(param)

    returned_loss = optimizer.step(closure)

    expected_loss, expected_grad = _dense_probe_average(theta0, 1e-3, seeds)
    exp_avg = (1.0 - 0.9) * expected_grad
    exp_avg_sq = (1.0 - 0.999) * expected_grad.square()
    expected_param = theta0 - 0.05 * exp_avg / (exp_avg_sq.sqrt() + 1e-10)

    assert calls["count"] == 2 * k
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_zo_adamm_replays_distinct_param_vectors_for_update(monkeypatch):
    _patch_randint(monkeypatch, [321])

    param_a = torch.nn.Parameter(torch.tensor([0.4, -0.1], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float32))
    optimizer = ZO_AdaMM([param_a, param_b], lr=0.05, eps=1e-3, k=1)

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    assert len(sample_calls) == 8
    assert not torch.allclose(sample_calls[0], sample_calls[1])
    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 4])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 6])


@pytest.mark.parametrize("k,seeds", [(1, [11]), (2, [11, 29])])
def test_jaguar_signsgd_matches_sparse_probe_average(monkeypatch, k, seeds):
    _patch_randint(monkeypatch, seeds)

    theta0 = torch.tensor([0.8, -0.4, 0.3, -0.6, 0.5, -0.7, 0.2, -0.1, 0.9, -0.8], dtype=torch.float32)
    initial_grad_accum = torch.tensor([0.6, -0.2, 0.7, -0.3, 0.8, -0.4, 0.9, -0.5, 1.0, -0.6], dtype=torch.float32)

    param = torch.nn.Parameter(theta0.clone())
    optimizer = Jaguar_SignSGD([param], lr=0.05, eps=1e-3, beta=0.9, k=k)
    optimizer.state[param]["step"] = 0
    optimizer.state[param]["grad_accum"] = initial_grad_accum.clone()

    closure, calls = _make_quadratic_closure(param)
    returned_loss = optimizer.step(closure)

    expected_loss, expected_sparse_grad, touched = _sparse_probe_average(theta0, 1e-3, seeds)
    expected_grad_accum = initial_grad_accum.clone()
    expected_grad_accum[touched] = 0.9 * expected_grad_accum[touched] + 0.1 * expected_sparse_grad[touched]
    expected_param = theta0 - 0.05 * torch.sign(expected_grad_accum)

    assert calls["count"] == 2 * k
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(optimizer.state[param]["grad_accum"], expected_grad_accum, atol=1e-6, rtol=1e-6)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        optimizer.state[param]["grad_accum"][~touched],
        initial_grad_accum[~touched],
    )


@pytest.mark.parametrize("k,seeds", [(1, [19]), (2, [19, 37])])
def test_sparse_jaguar_signsgd_matches_manual_probe_average(monkeypatch, k, seeds):
    _patch_randint(monkeypatch, [101, *seeds])

    theta0 = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    initial_grad_accum = torch.tensor([0.4, -0.6, 0.8], dtype=torch.float32)

    param = torch.nn.Parameter(theta0.clone())
    optimizer = Sparse_Jaguar_SignSGD(
        [param],
        lr=0.05,
        eps=1e-3,
        beta=0.9,
        params_ratio=1.0,
        k=k,
    )
    optimizer.state[param]["step"] = 0
    optimizer.state[param]["grad_accum"] = initial_grad_accum.clone()

    closure, calls = _make_quadratic_closure(param)
    returned_loss = optimizer.step(closure)

    expected_loss, expected_grad = _dense_probe_average(theta0, 1e-3, seeds)
    expected_grad_accum = 0.9 * initial_grad_accum + 0.1 * expected_grad
    expected_param = theta0 - 0.05 * torch.sign(expected_grad_accum)

    assert calls["count"] == 2 * k
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(optimizer.state[param]["grad_accum"], expected_grad_accum, atol=1e-6, rtol=1e-6)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_zo_rl_jaguar_replays_candidate_vectors_for_sparse_update(monkeypatch):
    _patch_randint(monkeypatch, [101, 11, 29])

    param_a = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.7, -0.1], dtype=torch.float32))
    optimizer = ZO_RL_Jaguar(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        beta=0.9,
        params_ratio=1.0,
        variance=1.0,
        lr_mu=0.1,
        k=2,
    )

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    seed_offsets = {11: 0, 29: 4}
    optimal_offset = seed_offsets[optimizer.zo_random_seed]

    assert len(sample_calls) == 20
    assert not torch.allclose(sample_calls[0], sample_calls[1])

    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset + 4], sample_calls[param_offset + 6])

        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[8 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[10 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[12 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[14 + param_offset])

        assert torch.allclose(sample_calls[param_offset], sample_calls[16 + param_offset])
        assert torch.allclose(sample_calls[4 + param_offset], sample_calls[18 + param_offset])


def test_zo_rl_jaguar_reuses_selected_params_across_k(monkeypatch):
    _patch_randint(monkeypatch, [101, 11, 29, 47])

    param_a0 = torch.tensor([0.2, -0.4], dtype=torch.float32)
    param_b0 = torch.tensor([0.7, -0.1], dtype=torch.float32)
    param_a = torch.nn.Parameter(param_a0.clone())
    param_b = torch.nn.Parameter(param_b0.clone())
    optimizer = ZO_RL_Jaguar(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        beta=0.9,
        params_ratio=0.5,
        variance=1.0,
        lr_mu=0.1,
        k=3,
    )

    selected_calls = {"count": 0}

    def fixed_selected_param_ids(params_ratio=0.1):
        selected_calls["count"] += 1
        return {id(param_a)}

    monkeypatch.setattr(optimizer, "_sample_selected_param_ids", fixed_selected_param_ids)

    closure_calls = {"count": 0}

    def closure():
        closure_calls["count"] += 1
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    assert selected_calls["count"] == 1
    assert closure_calls["count"] == 5
    assert torch.allclose(param_b.detach(), param_b0)


def test_zo_rl_sgd_replays_candidate_vectors_for_mu_update(monkeypatch):
    _patch_randint(monkeypatch, [11, 29])

    param_a = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.7, -0.1], dtype=torch.float32))
    optimizer = ZO_RL_SGD(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        momentum=0.0,
        variance=1.0,
        lr_mu=0.1,
        k=2,
    )

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    seed_offsets = {11: 0, 29: 4}
    optimal_offset = seed_offsets[optimizer.zo_random_seed]

    assert len(sample_calls) == 20
    assert not torch.allclose(sample_calls[0], sample_calls[1])

    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset + 4], sample_calls[param_offset + 6])

        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[8 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[10 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[12 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[14 + param_offset])

        assert torch.allclose(sample_calls[param_offset], sample_calls[16 + param_offset])
        assert torch.allclose(sample_calls[4 + param_offset], sample_calls[18 + param_offset])


def test_zo_rl_sgd_random_candidate_selection_uses_random_seed(monkeypatch):
    theta0 = torch.tensor([0.35, -0.25, 0.45], dtype=torch.float32)
    seeds = [11, 29]
    probe_results = [_dense_probe_single(theta0, 1e-3, seed) for seed in seeds]
    best_idx = min(range(len(seeds)), key=lambda idx: probe_results[idx][0].item())
    random_idx = 1 - best_idx
    expected_loss, expected_grad = probe_results[random_idx]

    _patch_randint(monkeypatch, [*seeds, random_idx])

    param = torch.nn.Parameter(theta0.clone())
    optimizer = ZO_RL_SGD(
        [param],
        lr=0.05,
        eps=1e-3,
        momentum=0.0,
        variance=1.0,
        lr_mu=0.0,
        k=2,
        candidate_selection_strategy="random",
    )
    closure, calls = _make_quadratic_closure(param)

    returned_loss = optimizer.step(closure)
    expected_param = theta0 - 0.05 * expected_grad

    assert calls["count"] == 4
    assert optimizer.zo_random_seed == seeds[random_idx]
    assert optimizer.zo_random_seed != seeds[best_idx]
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_zo_rl_sgd_mean_selection_averages_top_three_gradients(monkeypatch):
    theta0 = torch.tensor([0.25, -0.15, 0.4], dtype=torch.float32)
    seeds = [11, 29, 47, 53]
    probe_results = [_dense_probe_single(theta0, 1e-3, seed) for seed in seeds]
    ranked_indices = sorted(
        range(len(seeds)),
        key=lambda idx: probe_results[idx][0].item(),
    )
    selected_indices = ranked_indices[:3]
    expected_loss = torch.stack([probe_results[idx][0] for idx in selected_indices]).mean()
    expected_grad = torch.stack([probe_results[idx][1] for idx in selected_indices]).mean(dim=0)
    expected_seeds = [seeds[idx] for idx in selected_indices]

    _patch_randint(monkeypatch, seeds)

    param = torch.nn.Parameter(theta0.clone())
    optimizer = ZO_RL_SGD(
        [param],
        lr=0.05,
        eps=1e-3,
        momentum=0.0,
        variance=1.0,
        lr_mu=0.0,
        k=4,
        candidate_selection_strategy="mean",
        candidate_average_count=3,
    )
    closure, calls = _make_quadratic_closure(param)

    returned_loss = optimizer.step(closure)
    expected_param = theta0 - 0.05 * expected_grad

    assert calls["count"] == 10
    assert optimizer.selected_zo_random_seeds == expected_seeds
    assert torch.allclose(returned_loss, expected_loss)
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_zo_rl_adamm_replays_candidate_vectors_for_mu_update(monkeypatch):
    _patch_randint(monkeypatch, [13, 31])

    param_a = torch.nn.Parameter(torch.tensor([0.25, -0.35], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.6, -0.15], dtype=torch.float32))
    optimizer = ZO_RL_AdaMM(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        variance=1.0,
        lr_mu=0.1,
        k=2,
    )

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    seed_offsets = {13: 0, 31: 4}
    optimal_offset = seed_offsets[optimizer.zo_random_seed]

    assert len(sample_calls) == 20
    assert not torch.allclose(sample_calls[0], sample_calls[1])

    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset + 4], sample_calls[param_offset + 6])

        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[8 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[10 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[12 + param_offset])
        assert torch.allclose(sample_calls[optimal_offset + param_offset], sample_calls[14 + param_offset])

        assert torch.allclose(sample_calls[param_offset], sample_calls[16 + param_offset])
        assert torch.allclose(sample_calls[4 + param_offset], sample_calls[18 + param_offset])


def test_sparse_jaguar_signsgd_reuses_probe_vectors_for_update(monkeypatch):
    _patch_randint(monkeypatch, [101, 202])

    param_a = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.tensor([0.7, -0.1], dtype=torch.float32))
    optimizer = Sparse_Jaguar_SignSGD(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        beta=0.9,
        params_ratio=1.0,
        k=1,
    )

    sample_calls = []
    original_sample = optimizer.tensor_sampler.sample

    def recording_sample(param_shape, generator=None, sampler_type=None):
        z = original_sample(param_shape, generator=generator, sampler_type=sampler_type)
        sample_calls.append(z.detach().clone().cpu())
        return z

    monkeypatch.setattr(optimizer.tensor_sampler, "sample", recording_sample)

    def closure():
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    assert len(sample_calls) == 8
    assert not torch.allclose(sample_calls[0], sample_calls[1])
    for param_offset in (0, 1):
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 2])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 4])
        assert torch.allclose(sample_calls[param_offset], sample_calls[param_offset + 6])


def test_sparse_jaguar_signsgd_reuses_selected_params_across_k(monkeypatch):
    _patch_randint(monkeypatch, [101, 11, 29, 47])

    param_a0 = torch.tensor([0.2, -0.4], dtype=torch.float32)
    param_b0 = torch.tensor([0.7, -0.1], dtype=torch.float32)
    param_a = torch.nn.Parameter(param_a0.clone())
    param_b = torch.nn.Parameter(param_b0.clone())
    optimizer = Sparse_Jaguar_SignSGD(
        [param_a, param_b],
        lr=0.05,
        eps=1e-3,
        beta=0.9,
        params_ratio=0.5,
        k=3,
    )

    selected_calls = {"count": 0}

    def fixed_selected_param_ids(params_ratio=0.1):
        selected_calls["count"] += 1
        return {id(param_a)}

    monkeypatch.setattr(optimizer, "_sample_selected_param_ids", fixed_selected_param_ids)

    closure_calls = {"count": 0}

    def closure():
        closure_calls["count"] += 1
        return param_a.square().sum() + param_b.square().sum()

    optimizer.step(closure)

    assert selected_calls["count"] == 1
    assert closure_calls["count"] == 6
    assert not torch.allclose(param_a.detach(), param_a0)
    assert torch.allclose(param_b.detach(), param_b0)


def test_resolve_k_value_defaults():
    assert resolve_k_value("zo_sgd", None) == 1
    assert resolve_k_value("zo_adamm", None) == 1
    assert resolve_k_value("jaguar_signsgd", None) == 1
    assert resolve_k_value("sparse_jaguar_signsgd", None) == 1
    assert resolve_k_value("zo_rl_jaguar", None) == 10
    assert resolve_k_value("zo_rl_sgd", None) == 10
    assert resolve_k_value("zo_rl_adamm", None) == 10
    assert resolve_k_value("hizoo_rl", None) == 10
    assert resolve_k_value("mezo_svrg_rl", None) == 10
    assert resolve_k_value("zo_sgd", 7) == 7
