"""Surrogate gradients for the non-differentiable spike (Heaviside) function.

The QLIF neuron emits a hard 0/1 spike, whose derivative is zero almost
everywhere. Forward stays exact; backward substitutes a smooth surrogate so the
variational circuit parameters upstream still receive useful gradients.
"""
from __future__ import annotations

import torch


class _FastSigmoidSpike(torch.autograd.Function):
    """d/dx ~ 1 / (alpha*|x| + 1)^2  (Zenke & Ganguli, SuperSpike)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (ctx.alpha * x.abs() + 1.0) ** 2
        return grad_out * sg, None


class _AtanSpike(torch.autograd.Function):
    """d/dx ~ alpha / (2 * (1 + (pi/2 * alpha * x)^2))."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        sg = a / (2.0 * (1.0 + (torch.pi / 2.0 * a * x) ** 2))
        return grad_out * sg, None


class _SigmoidSpike(torch.autograd.Function):
    """d/dx ~ alpha * s * (1 - s), s = sigmoid(alpha * x)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(ctx.alpha * x)
        return grad_out * ctx.alpha * s * (1.0 - s), None


_FNS = {
    "fast_sigmoid": _FastSigmoidSpike,
    "atan": _AtanSpike,
    "sigmoid": _SigmoidSpike,
}


def spike_fn(x: torch.Tensor, kind: str = "fast_sigmoid", alpha: float = 5.0) -> torch.Tensor:
    """Heaviside(x) forward, smooth surrogate backward."""
    try:
        fn = _FNS[kind]
    except KeyError:
        raise ValueError(f"unknown surrogate '{kind}'; pick from {sorted(_FNS)}") from None
    return fn.apply(x, alpha)


def stochastic_spike(p: torch.Tensor) -> torch.Tensor:
    """Sample a spike from P(|1>) with a straight-through estimator.

    This is the *measurement-faithful* mode: on real hardware each timestep is a
    projective measurement, so the spike is a Bernoulli draw from the qubit's
    excited-state probability rather than a threshold on it.
    """
    s = torch.bernoulli(p.clamp(0.0, 1.0))
    return p + (s - p).detach()
