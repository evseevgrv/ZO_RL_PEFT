"""CPU checks for the learnable-mean ZO optimizers on a linear objective.

For f(x) = <g, x> the REINFORCE estimate of grad_mu E f(x + eps * v) is
eps * g in expectation, so mu must turn towards -g. The test also checks
that one step costs K + 1 forward passes and that the perturbations are
undone exactly when lr = 0.

Run: python test_mu_update.py
"""
import importlib.machinery
import sys
import types

import numpy as np
import torch

# Optimizers log to wandb when a run is active; a stub keeps the test offline.
_wandb = types.ModuleType("wandb")
_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
_wandb.run = None
_wandb.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", _wandb)

from optimizers import ZO_RL_SGD, ZO_RL_AdaMM  # noqa: E402

K = 5
STEPS = 400
SHAPES = [(7, 5), (11,), (3, 4)]


def make_problem(seed: int):
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(s)) for s in SHAPES]
    grads = [torch.randn(s) for s in SHAPES]
    calls = {"n": 0}

    def closure():
        calls["n"] += 1
        return sum((p * g).sum() for p, g in zip(params, grads))

    return params, grads, calls, closure


def cosine(a, b):
    a = torch.cat([t.flatten() for t in a])
    b = torch.cat([t.flatten() for t in b])
    return float(a @ b / (a.norm() * b.norm()))


def check(name, build):
    np.random.seed(0)
    params, grads, calls, closure = make_problem(0)
    x0 = [p.detach().clone() for p in params]
    opt = build(params)
    for _ in range(STEPS):
        opt.step(closure)
    per_step = calls["n"] / STEPS
    drift = max(float((p - q).abs().max()) for p, q in zip(params, x0))
    mus = [opt.state[p]["mu"] for p in params]
    cos = cosine(mus, [-g for g in grads])
    print(f"{name}: forwards/step={per_step:.0f} drift={drift:.1e} cos(mu,-grad)={cos:.3f}")
    assert per_step == K + 1, per_step
    assert drift < 1e-5, drift
    assert cos > 0.9, cos


if __name__ == "__main__":
    common = dict(lr=0.0, eps=1e-2, k=K, variance=1.0, lr_mu=1e-2)
    check("ZO_RL_SGD", lambda ps: ZO_RL_SGD(ps, momentum=0.0, **common))
    check("ZO_RL_AdaMM", lambda ps: ZO_RL_AdaMM(ps, **common))
    print("ok")
