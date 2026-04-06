import copy
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

from optimizers.mezo_svrg import MeZO_SVRG
from optimizers.mezo_svrg_rl import MeZO_SVRG_RL


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


def _copy_optimizer_state(src_optimizer, src_param, dst_optimizer, dst_param):
    dst_param.data.copy_(src_param.data)
    dst_optimizer.global_step = src_optimizer.global_step
    dst_optimizer.projected_grad = src_optimizer.projected_grad

    dst_state = dst_optimizer.state[dst_param]
    dst_state.clear()

    for key, value in src_optimizer.state[src_param].items():
        if torch.is_tensor(value):
            dst_state[key] = value.detach().clone()
        else:
            dst_state[key] = copy.deepcopy(value)


def test_mezo_svrg_rl_full_step_populates_snapshot_and_full_grad(monkeypatch):
    _patch_randint(monkeypatch, [13, 29])

    theta0 = torch.tensor([0.35, -0.25, 0.5], dtype=torch.float32)
    param = torch.nn.Parameter(theta0.clone())
    optimizer = MeZO_SVRG_RL(
        [param],
        lr=0.1,
        eps=1e-3,
        q=2,
        k=2,
        variance=1.0,
        lr_mu=0.0,
    )
    closure, calls = _make_quadratic_closure(param)

    optimizer.step({"mini": closure, "full": closure})

    state = optimizer.state[param]

    assert calls["count"] == 4
    assert torch.allclose(state["snapshot_param_cpu"], theta0)
    assert torch.linalg.norm(state["full_grad_cpu"]).item() > 0
    assert torch.allclose(param.detach(), theta0 - 0.1 * state["full_grad_cpu"])


def test_mezo_svrg_rl_matches_mezo_svrg_when_k1_mu0(monkeypatch):
    seeds = [17, 31, 43]
    theta0 = torch.tensor([0.4, -0.15, 0.2], dtype=torch.float32)

    base_param = torch.nn.Parameter(theta0.clone())
    base_optimizer = MeZO_SVRG([base_param], lr=0.05, eps=1e-3, q=2)
    base_closure, _ = _make_quadratic_closure(base_param)
    _patch_randint(monkeypatch, seeds)
    base_loss_1 = base_optimizer.step({"mini": base_closure, "full": base_closure})
    base_loss_2 = base_optimizer.step({"mini": base_closure, "full": base_closure})

    rl_param = torch.nn.Parameter(theta0.clone())
    rl_optimizer = MeZO_SVRG_RL(
        [rl_param],
        lr=0.05,
        eps=1e-3,
        q=2,
        k=1,
        variance=1.0,
        lr_mu=0.0,
    )
    rl_closure, _ = _make_quadratic_closure(rl_param)
    _patch_randint(monkeypatch, seeds)
    rl_loss_1 = rl_optimizer.step({"mini": rl_closure, "full": rl_closure})
    rl_loss_2 = rl_optimizer.step({"mini": rl_closure, "full": rl_closure})

    assert torch.allclose(base_loss_1, rl_loss_1)
    assert torch.allclose(base_loss_2, rl_loss_2)
    assert torch.allclose(base_param.detach(), rl_param.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        base_optimizer.state[base_param]["snapshot_param_cpu"],
        rl_optimizer.state[rl_param]["snapshot_param_cpu"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        base_optimizer.state[base_param]["full_grad_cpu"],
        rl_optimizer.state[rl_param]["full_grad_cpu"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_mezo_svrg_rl_minibatch_x_update_is_independent_of_mu_update(monkeypatch):
    theta0 = torch.tensor([0.2, -0.45, 0.7], dtype=torch.float32)

    reference_param = torch.nn.Parameter(theta0.clone())
    reference_optimizer = MeZO_SVRG_RL(
        [reference_param],
        lr=0.05,
        eps=1e-3,
        q=2,
        k=2,
        variance=1.0,
        lr_mu=0.0,
    )
    reference_closure, _ = _make_quadratic_closure(reference_param)
    _patch_randint(monkeypatch, [7, 19])
    reference_optimizer.step({"mini": reference_closure, "full": reference_closure})

    param_no_mu = torch.nn.Parameter(reference_param.detach().clone())
    optimizer_no_mu = MeZO_SVRG_RL(
        [param_no_mu],
        lr=0.05,
        eps=1e-3,
        q=2,
        k=2,
        variance=1.0,
        lr_mu=0.0,
    )
    _copy_optimizer_state(
        reference_optimizer, reference_param, optimizer_no_mu, param_no_mu
    )

    param_with_mu = torch.nn.Parameter(reference_param.detach().clone())
    optimizer_with_mu = MeZO_SVRG_RL(
        [param_with_mu],
        lr=0.05,
        eps=1e-3,
        q=2,
        k=2,
        variance=1.0,
        lr_mu=0.3,
    )
    _copy_optimizer_state(
        reference_optimizer, reference_param, optimizer_with_mu, param_with_mu
    )

    minibatch_seeds = [101, 131, 151, 181]
    no_mu_closure, _ = _make_quadratic_closure(param_no_mu)
    _patch_randint(monkeypatch, minibatch_seeds)
    loss_no_mu = optimizer_no_mu.step({"mini": no_mu_closure, "full": no_mu_closure})

    with_mu_closure, _ = _make_quadratic_closure(param_with_mu)
    _patch_randint(monkeypatch, minibatch_seeds)
    loss_with_mu = optimizer_with_mu.step(
        {"mini": with_mu_closure, "full": with_mu_closure}
    )

    assert torch.allclose(loss_no_mu, loss_with_mu)
    assert torch.allclose(
        param_no_mu.detach(), param_with_mu.detach(), atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(
        optimizer_no_mu.state[param_no_mu]["mu"],
        optimizer_with_mu.state[param_with_mu]["mu"],
    )


def test_mezo_svrg_rl_supports_use_grad_first(monkeypatch):
    _patch_randint(monkeypatch, [23])

    param = torch.nn.Parameter(torch.tensor([0.15, -0.3], dtype=torch.float32))
    param.grad = torch.tensor([0.4, -0.2], dtype=torch.float32)

    optimizer = MeZO_SVRG_RL(
        [param],
        lr=0.1,
        eps=1e-3,
        q=2,
        k=1,
        variance=1.0,
        lr_mu=0.0,
        use_grad_first=True,
    )
    closure, calls = _make_quadratic_closure(param)

    optimizer.step({"mini": closure, "full": closure})

    assert calls["count"] == 3
    assert torch.allclose(optimizer.state[param]["mu_old"], param.grad)
    assert torch.allclose(optimizer.state[param]["mu"], param.grad)
