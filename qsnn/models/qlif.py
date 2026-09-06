"""Quantum Leaky Integrate-and-Fire (QLIF) neuron layer.

One qubit == one neuron. The neuron's *membrane potential* is carried by the
qubit's Bloch-sphere polar angle, and the firing probability is literally the
excited-state population P(|1>).

Per timestep t, for a population of N qubits:

  1. Integrate (classical, cheap):
         U_t = beta * U_{t-1} * (1 - S_{t-1})  +  I_t
     where I_t is the synaptic current from the previous layer and beta is the
     leak. The reset term implements refractoriness.

  2. Prepare (quantum): map the membrane to a bounded rotation angle and encode
     it on the qubit,
         theta_t = pi * sigmoid(g * U_t + b)   ->   RY(theta_t) | 0 >
     so that  P(|1>) = sin^2(theta_t / 2)  is *strictly increasing* in U_t over
     the whole real line. This matters: a symmetric map such as pi*tanh(U) makes
     a strongly inhibited neuron (U << 0) fire exactly as hard as a strongly
     excited one (U >> 0), which cancels the sign of every synaptic weight and
     stops the network learning. The default (g=2, b=-2) puts the resting
     potential U=0 at P(|1>) ~ 0.03 (quiet) and saturates near U=2.

  3. Couple (quantum): a trainable entangling ansatz mixes the population. This
     is the part with no classical analogue - neurons share amplitude, not just
     a weight matrix, which is where the model's expressivity over a classical
     LIF layer comes from.

  4. Measure & fire:
         P(|1>_i) = (1 - <Z_i>) / 2
         S_t^i    = Heaviside(P(|1>_i) - v_th)        [surrogate gradient]
     Measurement collapses the qubit, so the next timestep re-prepares from the
     classical membrane - the "measure-and-prepare" scheme that makes QLIF
     shallow enough for NISQ hardware: circuit depth is O(ansatz_layers) per
     timestep and does **not** grow with T.

Depth is therefore constant in the sequence length, which is exactly why this
survives on noisy hardware where a QLSTM/QCNN of equivalent temporal reach does
not.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn

from ..config import ModelConfig, NoiseConfig
from ..noise import make_device, resolve_diff_method
from .surrogate import spike_fn, stochastic_spike


# --------------------------------------------------------------------------- #
def _ansatz(weights: torch.Tensor, wires: List[int], kind: str) -> None:
    """Trainable entangling block. weights shape: [layers, n_wires, 3|1]."""
    n = len(wires)
    if kind == "strong":
        qml.StronglyEntanglingLayers(weights, wires=wires)
        return
    if kind == "basic":
        qml.BasicEntanglerLayers(weights, wires=wires)
        return
    if kind == "ring":
        # Hardware-efficient: single-qubit rotations + nearest-neighbour ring.
        # Cheapest to transpile onto heavy-hex IBM topologies.
        for layer in range(weights.shape[0]):
            for i, w in enumerate(wires):
                qml.RY(weights[layer, i, 0], wires=w)
                qml.RZ(weights[layer, i, 1], wires=w)
            if n > 1:
                for i in range(n):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
        return
    raise ValueError(f"unknown ansatz '{kind}'")


def _weight_shape(kind: str, layers: int, n_wires: int) -> Tuple[int, ...]:
    if kind == "strong":
        return (layers, n_wires, 3)
    if kind == "basic":
        return (layers, n_wires)
    if kind == "ring":
        return (layers, n_wires, 2)
    raise ValueError(f"unknown ansatz '{kind}'")


# --------------------------------------------------------------------------- #
class QuantumLIFLayer(nn.Module):
    """A population of ``n_qubits`` QLIF neurons, unrolled over T timesteps."""

    def __init__(
        self,
        n_qubits: int,
        ansatz_layers: int = 2,
        ansatz: str = "strong",
        beta: float = 0.9,
        learn_beta: bool = True,
        threshold: float = 0.55,
        learn_threshold: bool = False,
        reset: str = "subtract",
        angle_map: str = "sigmoid",
        angle_gain: float = 2.0,
        angle_bias: float = -2.0,
        surrogate: str = "fast_sigmoid",
        surrogate_alpha: float = 5.0,
        diff_method: str = "backprop",
        noise: Optional[NoiseConfig] = None,
        stochastic: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.reset = reset
        if angle_map not in ("sigmoid", "tanh"):
            raise ValueError(f"unknown angle_map '{angle_map}'")
        self.angle_map = angle_map
        self.angle_gain = float(angle_gain)
        self.angle_bias = float(angle_bias)
        self.surrogate = surrogate
        self.surrogate_alpha = surrogate_alpha
        self.stochastic = stochastic
        self.ansatz = ansatz
        self.ansatz_layers = ansatz_layers

        # ---- membrane parameters ------------------------------------------
        # beta and v_th are stored unconstrained and squashed in forward(), so
        # they stay in (0,1) under unconstrained optimisers.
        b0 = math.log(beta / (1.0 - beta))
        t0 = math.log(threshold / (1.0 - threshold))
        self.beta_logit = nn.Parameter(torch.tensor(b0), requires_grad=learn_beta)
        self.thr_logit = nn.Parameter(torch.tensor(t0), requires_grad=learn_threshold)

        # ---- quantum block -------------------------------------------------
        self.dev, self._noise_op, backprop_ok = make_device(self.n_qubits, noise)
        self.diff_method = resolve_diff_method(diff_method, backprop_ok)
        self.noisy = bool(noise and noise.enabled)

        wires = list(range(self.n_qubits))
        noise_op = self._noise_op
        ansatz_kind = ansatz

        @qml.qnode(self.dev, interface="torch", diff_method=self.diff_method)
        def circuit(inputs, weights):
            # (2) prepare: membrane angle -> qubit polar angle
            for i in wires:
                qml.RY(inputs[..., i], wires=i)
            noise_op(wires)
            # (3) couple: trainable entangling ansatz
            _ansatz(weights, wires, ansatz_kind)
            noise_op(wires)
            # (4) measure
            return [qml.expval(qml.PauliZ(i)) for i in wires]

        self.circuit = circuit
        shape = _weight_shape(ansatz, ansatz_layers, self.n_qubits)
        g = torch.Generator().manual_seed(seed)
        # Small init keeps the circuit near identity -> mitigates barren plateaus.
        init = 0.1 * torch.randn(shape, generator=g)
        self.qweights = nn.Parameter(init)

        self.register_buffer("_last_spike_rate", torch.zeros(1), persistent=False)

    # ------------------------------------------------------------------ props
    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    @property
    def v_th(self) -> torch.Tensor:
        return torch.sigmoid(self.thr_logit)

    # ---------------------------------------------------------------- circuit
    def membrane_to_angle(self, mem: torch.Tensor) -> torch.Tensor:
        """Membrane potential -> Bloch polar angle in [0, pi].

        'sigmoid' (default) is strictly monotone, so P(|1>) increases with the
        membrane over the whole real line and inhibition genuinely suppresses
        firing. 'tanh' reproduces the symmetric map and is kept only as an
        ablation to demonstrate why monotonicity is required.
        """
        if self.angle_map == "tanh":
            return math.pi * torch.tanh(mem)
        return math.pi * torch.sigmoid(self.angle_gain * mem + self.angle_bias)

    def _p_excited(self, theta: torch.Tensor) -> torch.Tensor:
        """theta [B, n_qubits] -> P(|1>) [B, n_qubits]."""
        out = self.circuit(theta, self.qweights)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=-1)
        z = out.to(theta.dtype)
        if z.dim() == 1:                       # unbatched device return
            z = z.unsqueeze(0)
        return (1.0 - z) * 0.5

    # ---------------------------------------------------------------- forward
    def forward(
        self, current: torch.Tensor, return_states: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``current``: [B, T, n_qubits] synaptic input -> spikes [B, T, n_qubits]."""
        if current.dim() != 3:
            raise ValueError(f"expected [B, T, N], got {tuple(current.shape)}")
        B, T, N = current.shape
        if N != self.n_qubits:
            raise ValueError(f"layer has {self.n_qubits} qubits, got {N} channels")

        beta, v_th = self.beta, self.v_th
        mem = torch.zeros(B, N, dtype=current.dtype, device=current.device)
        spk = torch.zeros_like(mem)

        spikes: List[torch.Tensor] = []
        mems: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []

        for t in range(T):
            # (1) leaky integration with refractory reset
            if self.reset == "zero":
                mem = beta * mem * (1.0 - spk) + current[:, t]
            elif self.reset == "subtract":
                mem = beta * mem + current[:, t] - spk * v_th
            else:
                mem = beta * mem + current[:, t]

            # (2)+(3)+(4) prepare -> entangle -> measure
            theta = self.membrane_to_angle(mem)
            p1 = self._p_excited(theta)

            spk = (stochastic_spike(p1) if self.stochastic
                   else spike_fn(p1 - v_th, self.surrogate, self.surrogate_alpha))

            spikes.append(spk)
            if return_states:
                mems.append(mem)
                probs.append(p1)

        out = torch.stack(spikes, dim=1)
        self._last_spike_rate = out.detach().mean().reshape(1)
        if not return_states:
            return out
        return out, {
            "mem": torch.stack(mems, dim=1),
            "p_excited": torch.stack(probs, dim=1),
            "final_mem": mem,
        }

    # ------------------------------------------------------------------ specs
    def circuit_specs(self) -> Dict[str, int | float]:
        """Depth / gate counts of ONE timestep (depth is constant in T)."""
        dummy = torch.zeros(1, self.n_qubits)
        depth = n_gates = two_q = 0
        try:
            # level="device" reports the DECOMPOSED circuit, i.e. the gates that
            # actually run on hardware. Without it, a template like
            # StronglyEntanglingLayers is counted as a single gate of depth 1.
            try:
                spec = qml.specs(self.circuit, level="device")(dummy, self.qweights)
            except TypeError:                               # older PennyLane
                spec = qml.specs(self.circuit)(dummy, self.qweights)
            # PennyLane >=0.42 returns a CircuitSpecs object; older versions a dict.
            raw = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
            res = raw.get("resources", raw)
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            depth = int(res.get("depth", 0) or 0)
            n_gates = int(res.get("num_gates", res.get("num_operations", 0)) or 0)
            # gate_sizes maps arity -> count, so "2" is the two-qubit gate count
            sizes = {str(k): v for k, v in (res.get("gate_sizes", {}) or {}).items()}
            if sizes:
                two_q = int(sizes.get("2", 0))
            else:
                types = res.get("gate_types", {}) or {}
                two_q = sum(v for k, v in types.items()
                            if k in ("CNOT", "CZ", "CRX", "CRY", "CRZ", "ArbitraryUnitary"))
        except Exception as exc:                            # pragma: no cover
            print(f"[qlif] circuit_specs unavailable: {type(exc).__name__}: {exc}")
        return {
            "n_qubits": self.n_qubits,
            "depth_per_timestep": int(depth),
            "gates_per_timestep": int(n_gates),
            "two_qubit_gates_per_timestep": int(two_q),
            "quantum_params": int(self.qweights.numel()),
            "ansatz": self.ansatz,
            "ansatz_layers": self.ansatz_layers,
            "diff_method": self.diff_method,
            "noisy": self.noisy,
        }

    def extra_repr(self) -> str:
        return (f"n_qubits={self.n_qubits}, ansatz={self.ansatz}x{self.ansatz_layers}, "
                f"beta={float(self.beta):.3f}, v_th={float(self.v_th):.3f}, "
                f"reset={self.reset}, angle_map={self.angle_map}, "
                f"diff={self.diff_method}, noisy={self.noisy}")


# --------------------------------------------------------------------------- #
class ClassicalLIFLayer(nn.Module):
    """Ablation control: identical dynamics, classical sigmoid instead of the
    qubit. Isolates what the quantum block actually buys."""

    def __init__(self, n_neurons: int, beta: float = 0.9, threshold: float = 0.55,
                 learn_beta: bool = True, reset: str = "subtract",
                 surrogate: str = "fast_sigmoid", surrogate_alpha: float = 5.0):
        super().__init__()
        self.n_qubits = n_neurons
        self.reset = reset
        self.surrogate = surrogate
        self.surrogate_alpha = surrogate_alpha
        b0 = math.log(beta / (1.0 - beta))
        self.beta_logit = nn.Parameter(torch.tensor(b0), requires_grad=learn_beta)
        self.register_buffer("v_th_buf", torch.tensor(threshold))
        self.mix = nn.Linear(n_neurons, n_neurons, bias=False)

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, current: torch.Tensor, return_states: bool = False):
        B, T, N = current.shape
        mem = torch.zeros(B, N, dtype=current.dtype, device=current.device)
        spk = torch.zeros_like(mem)
        outs, mems, probs = [], [], []
        for t in range(T):
            if self.reset == "zero":
                mem = self.beta * mem * (1.0 - spk) + current[:, t]
            else:
                mem = self.beta * mem + current[:, t] - spk * self.v_th_buf
            theta = math.pi * torch.sigmoid(2.0 * mem - 2.0)
            p1 = torch.sin(self.mix(theta) / 2.0) ** 2
            spk = spike_fn(p1 - self.v_th_buf, self.surrogate, self.surrogate_alpha)
            outs.append(spk)
            if return_states:
                mems.append(mem)
                probs.append(p1)
        out = torch.stack(outs, dim=1)
        if not return_states:
            return out
        return out, {"mem": torch.stack(mems, 1), "p_excited": torch.stack(probs, 1),
                     "final_mem": mem}

    def circuit_specs(self) -> Dict[str, int | float]:
        return {"n_qubits": 0, "depth_per_timestep": 0, "gates_per_timestep": 0,
                "two_qubit_gates_per_timestep": 0,
                "quantum_params": 0, "ansatz": "classical-lif",
                "ansatz_layers": 0, "diff_method": "backprop", "noisy": False}
