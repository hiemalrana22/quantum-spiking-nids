"""Feature vector -> spike train.

This is the step that makes the model *spiking*. A flow is represented as a
binary tensor ``[B, T, F]`` instead of a static vector ``[B, F]``, so the
network integrates evidence over T discrete timesteps the way a QLIF neuron
does on hardware.

Schemes
-------
rate     Bernoulli(x_i) per timestep. Feature magnitude -> firing *rate*.
latency  Time-to-first-spike: a large feature fires early, a small one late
         (or never). One spike per feature - the sparsest, cheapest code.
phase    Deterministic ramp/threshold code; no sampling, so gradients through
         the encoder are stable and runs are exactly reproducible.
delta    Fires when the running reconstruction error exceeds a threshold
         (sigma-delta / event-camera style). Sparse and precision-preserving.
repeat   No spikes at all - repeats the analogue vector T times. Ablation
         control that isolates how much the spike code itself contributes.
window   Sequence mode: T is real time (T consecutive flows), each carrying its
         own analogue feature vector. Only usable on timestamped datasets
         (CIC-IDS2017), and built by :func:`make_flow_windows`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..config import EncodingConfig

SCHEMES = ("rate", "latency", "phase", "delta", "repeat")


# --------------------------------------------------------------------------- #
def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def rate_code(x: torch.Tensor, T: int, gain: float = 1.0,
              deterministic: bool = False,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """x in [0,1] -> Bernoulli spikes with p = clamp(gain * x)."""
    p = torch.clamp(x * gain, 0.0, 1.0).unsqueeze(1).expand(-1, T, -1)
    if deterministic:
        # evenly spaced deterministic spikes with the same mean rate
        ramp = (torch.arange(T, device=x.device, dtype=x.dtype) + 0.5) / T
        return (p >= ramp.view(1, T, 1)).float()
    noise = torch.rand(p.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (noise < p).float()


def latency_code(x: torch.Tensor, T: int, tau: float = 3.0,
                 gain: float = 1.0) -> torch.Tensor:
    """Time-to-first-spike. t_i = tau * ln(1 / (x_i + eps)); one spike each."""
    eps = 1e-4
    xv = torch.clamp(x * gain, 0.0, 1.0)
    t = tau * torch.log(1.0 / (xv + eps))
    t = torch.round(torch.clamp(t, 0.0, float(T)))                 # T == "never"
    idx = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1)
    return (idx == t.unsqueeze(1)).float()


def phase_code(x: torch.Tensor, T: int, gain: float = 1.0) -> torch.Tensor:
    """Deterministic thermometer/ramp code - no RNG, exactly reproducible."""
    xv = torch.clamp(x * gain, 0.0, 1.0)
    ramp = (torch.arange(T, device=x.device, dtype=x.dtype) + 0.5) / T
    return (xv.unsqueeze(1) >= ramp.view(1, T, 1)).float()


def delta_code(x: torch.Tensor, T: int, gain: float = 1.0,
               threshold: float = 0.25) -> torch.Tensor:
    """Sigma-delta: emit a spike whenever accumulated residual crosses theta."""
    xv = torch.clamp(x * gain, 0.0, 1.0)
    acc = torch.zeros_like(xv)
    out = []
    step = xv / max(T, 1) * T ** 0.5
    for _ in range(T):
        acc = acc + step
        s = (acc >= threshold).float()
        acc = acc - s * threshold
        out.append(s)
    return torch.stack(out, dim=1)


def repeat_code(x: torch.Tensor, T: int, gain: float = 1.0) -> torch.Tensor:
    """Ablation control: analogue current held constant for T steps."""
    return torch.clamp(x * gain, 0.0, 1.0).unsqueeze(1).expand(-1, T, -1).contiguous()


# --------------------------------------------------------------------------- #
class SpikeEncoder:
    """Stateless callable: ``[B, F] -> [B, T, F]``.

    Kept as a module-free object so it can run inside a DataLoader worker or on
    the GPU inside the training loop, whichever is cheaper.
    """

    def __init__(self, cfg: EncodingConfig):
        if cfg.scheme not in SCHEMES:
            raise ValueError(f"unknown spike scheme '{cfg.scheme}'; pick from {SCHEMES}")
        self.cfg = cfg
        self._gen: Optional[torch.Generator] = None

    def _generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.cfg.scheme != "rate" or self.cfg.deterministic:
            return None
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.cfg.seed)
        return self._gen

    def __call__(self, x) -> torch.Tensor:
        x = _as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3:                       # already a sequence -> window mode
            return x
        c = self.cfg
        if c.scheme == "rate":
            return rate_code(x, c.timesteps, c.gain, c.deterministic,
                             self._generator(x.device))
        if c.scheme == "latency":
            return latency_code(x, c.timesteps, c.tau, c.gain)
        if c.scheme == "phase":
            return phase_code(x, c.timesteps, c.gain)
        if c.scheme == "delta":
            return delta_code(x, c.timesteps, c.gain)
        return repeat_code(x, c.timesteps, c.gain)

    # ------------------------------------------------------------------ info
    def sparsity(self, x) -> float:
        """Mean fraction of active (spiking) input units - the quantity that
        drives the energy-efficiency argument for spiking hardware."""
        return float(self(x).mean().item())

    def describe(self) -> str:
        c = self.cfg
        return (f"SpikeEncoder(scheme={c.scheme}, T={c.timesteps}, gain={c.gain}"
                + (f", tau={c.tau}" if c.scheme == "latency" else "")
                + (", deterministic" if c.deterministic else "") + ")")


# --------------------------------------------------------------------------- #
def make_flow_windows(
    X: np.ndarray, y: np.ndarray, T: int, stride: int = 1,
    label_rule: str = "any",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sequence mode for timestamped datasets (CIC-IDS2017).

    Slides a length-T window over consecutive flows so the QLIF membrane
    integrates over *real* traffic time rather than over an encoding axis.
    Rows must already be sorted by timestamp.

    ``label_rule``: 'any'  -> window is an attack if it contains one (recall-first)
                    'last' -> label of the final flow (online-detection framing)
                    'majority'
    """
    if len(X) < T:
        raise ValueError(f"need at least T={T} rows, got {len(X)}")
    n = (len(X) - T) // stride + 1
    idx = np.arange(T)[None, :] + np.arange(0, n * stride, stride)[:, None]
    Xw = X[idx]                                                   # [n, T, F]
    yw_all = y[idx]                                               # [n, T]
    if label_rule == "any":
        yw = (yw_all > 0).any(axis=1).astype(np.int64)
    elif label_rule == "last":
        yw = yw_all[:, -1].astype(np.int64)
    elif label_rule == "majority":
        yw = (yw_all > 0).mean(axis=1).round().astype(np.int64)
    else:
        raise ValueError(f"unknown label_rule '{label_rule}'")
    return Xw.astype(np.float32), yw
