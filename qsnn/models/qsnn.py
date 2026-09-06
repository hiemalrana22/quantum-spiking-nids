"""QSNN-NIDS: the full hybrid quantum-classical spiking intrusion detector.

    flow features x in [0,1]^F
        -> SpikeEncoder                    [B, T, F]   (rate / latency / phase / delta)
        -> Linear synapse  F -> Q          [B, T, Q]   (classical, trainable)
        -> QuantumLIFLayer  (Q qubits)     [B, T, Q]   spikes
        -> Linear synapse  Q -> Q          [B, T, Q]   (deep variant)
        -> QuantumLIFLayer  (Q qubits)     [B, T, Q]   spikes
        -> readout (spike rate + final membrane)
        -> Linear -> logits                [B, C]

Circuit depth is O(ansatz_layers) and independent of T, which is the property
that keeps it runnable on NISQ hardware.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config, EncodingConfig, ModelConfig, NoiseConfig
from ..data.spike_encoding import SpikeEncoder
from .qlif import ClassicalLIFLayer, QuantumLIFLayer


class QSNN(nn.Module):
    """Quantum Spiking Neural Network for network intrusion detection."""

    def __init__(
        self,
        n_features: int,
        model_cfg: ModelConfig,
        encoding_cfg: EncodingConfig,
        noise_cfg: Optional[NoiseConfig] = None,
        classical_lif: bool = False,
        stochastic: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.cfg = model_cfg
        self.enc_cfg = encoding_cfg
        self.encoder = SpikeEncoder(encoding_cfg)
        self.T = encoding_cfg.timesteps
        Q = model_cfg.n_qubits

        # ---- synaptic (classical) weights, shared across timesteps ---------
        dims = [n_features] + [Q] * model_cfg.n_qlif_layers
        self.synapses = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(model_cfg.n_qlif_layers)]
        )
        for lin in self.synapses:
            nn.init.xavier_uniform_(lin.weight, gain=0.8)
            nn.init.zeros_(lin.bias)

        # ---- quantum spiking populations -----------------------------------
        def make_layer(idx: int):
            if classical_lif:
                return ClassicalLIFLayer(
                    Q, beta=model_cfg.beta, threshold=model_cfg.threshold,
                    reset=model_cfg.reset, surrogate=model_cfg.surrogate,
                    surrogate_alpha=model_cfg.surrogate_alpha)
            return QuantumLIFLayer(
                n_qubits=Q,
                ansatz_layers=model_cfg.ansatz_layers,
                ansatz=model_cfg.ansatz,
                beta=model_cfg.beta,
                learn_beta=model_cfg.learn_beta,
                threshold=model_cfg.threshold,
                learn_threshold=model_cfg.learn_threshold,
                reset=model_cfg.reset,
                angle_map=model_cfg.angle_map,
                angle_gain=model_cfg.angle_gain,
                angle_bias=model_cfg.angle_bias,
                surrogate=model_cfg.surrogate,
                surrogate_alpha=model_cfg.surrogate_alpha,
                diff_method=model_cfg.diff_method,
                noise=noise_cfg,
                stochastic=stochastic,
                seed=1234 + idx,
            )

        self.qlif_layers = nn.ModuleList(
            [make_layer(i) for i in range(model_cfg.n_qlif_layers)]
        )

        # ---- readout --------------------------------------------------------
        # 'rate'  : mean spike count over T - exactly what hardware can observe,
        #           but quantised to T+1 levels.
        # 'mem'   : final membrane potential.
        # 'prob'  : mean P(|1>) - the infinite-shot limit of the spike rate;
        #           simulator-only, far lower variance, so it trains faster.
        # 'rate_mem' / 'rate_mem_prob': concatenations of the above.
        readout_dim = {
            "rate": Q, "mem": Q, "prob": Q,
            "rate_mem": 2 * Q, "rate_prob": 2 * Q, "rate_mem_prob": 3 * Q,
        }[model_cfg.readout]
        # BatchNorm, not LayerNorm: LayerNorm subtracts the per-sample mean and
        # would delete the population firing rate, which is itself the signal.
        self.norm = nn.BatchNorm1d(readout_dim)
        self.drop = nn.Dropout(model_cfg.dropout)
        self.head = nn.Linear(readout_dim, model_cfg.n_classes)
        nn.init.zeros_(self.head.bias)

        self._spike_rates: List[float] = []

    # ------------------------------------------------------------------ utils
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, F] -> [B, T, F]. Pass-through if already a [B, T, F] window."""
        return x if x.dim() == 3 else self.encoder(x)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        spikes_in = self.encode(x).to(next(self.parameters()).dtype)

        h = spikes_in
        aux: Dict[str, torch.Tensor] = {}
        self._spike_rates = []
        states: Dict[str, torch.Tensor] = {}

        for i, (syn, qlif) in enumerate(zip(self.synapses, self.qlif_layers)):
            cur = syn(h)                                   # [B, T, Q]
            h, states = qlif(cur, return_states=True)       # [B, T, Q]
            self._spike_rates.append(float(h.detach().mean()))
            if return_aux:
                aux[f"spikes_l{i}"] = h
                aux[f"mem_l{i}"] = states["mem"]

        rate = h.mean(dim=1)                                # [B, Q] firing rate
        mem = torch.tanh(states["final_mem"])               # [B, Q] final potential
        prob = states["p_excited"].mean(dim=1)              # [B, Q] mean P(|1>)
        parts = {"rate": rate, "mem": mem, "prob": prob}
        feats = torch.cat([parts[k] for k in self.cfg.readout.split("_")], dim=-1)

        logits = self.head(self.drop(self.norm(feats)))
        if return_aux:
            aux["input_spikes"] = spikes_in
            aux["readout"] = feats
            return logits, aux
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=-1)

    # ---------------------------------------------------------------- reports
    @property
    def spike_rate(self) -> float:
        """Mean fraction of neurons firing - the sparsity/energy proxy."""
        return float(sum(self._spike_rates) / max(len(self._spike_rates), 1))

    def resource_report(self) -> Dict[str, object]:
        specs = [l.circuit_specs() for l in self.qlif_layers]
        n_class = sum(p.numel() for n, p in self.named_parameters()
                      if "qweights" not in n and p.requires_grad)
        n_quant = sum(p.numel() for n, p in self.named_parameters()
                      if "qweights" in n)
        return {
            "n_qubits": self.cfg.n_qubits,
            "n_qlif_layers": self.cfg.n_qlif_layers,
            "timesteps": self.T,
            "spike_scheme": self.enc_cfg.scheme,
            "classical_params": int(n_class),
            "quantum_params": int(n_quant),
            "total_params": int(n_class + n_quant),
            "depth_per_timestep": max((s["depth_per_timestep"] for s in specs), default=0),
            "total_circuit_depth": sum(s["depth_per_timestep"] for s in specs) * self.T,
            "two_qubit_gates_per_timestep": sum(
                s["two_qubit_gates_per_timestep"] for s in specs),
            "circuit_evaluations_per_sample": self.T * self.cfg.n_qlif_layers,
            "noisy": any(s["noisy"] for s in specs),
            "diff_method": specs[0]["diff_method"] if specs else "n/a",
            "per_layer": specs,
        }

    @torch.no_grad()
    def benchmark_latency(self, x: torch.Tensor, repeats: int = 3) -> Dict[str, float]:
        """Wall-clock inference latency on the current device (simulator)."""
        self.eval()
        self(x[:1])                                          # warm up
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self(x)
            ts.append(time.perf_counter() - t0)
        best = min(ts)
        return {"batch_size": int(len(x)),
                "latency_ms_total": best * 1e3,
                "latency_ms_per_sample": best * 1e3 / max(len(x), 1),
                "throughput_flows_per_s": len(x) / best}

    def describe(self) -> str:
        r = self.resource_report()
        lines = [
            "QSNN-NIDS",
            f"  input features        : {self.n_features}",
            f"  spike encoding        : {self.encoder.describe()}",
            f"  qubits                : {r['n_qubits']}  ({r['n_qlif_layers']} QLIF layers)",
            f"  circuit depth / step  : {r['depth_per_timestep']}  "
            f"(2q gates: {r['two_qubit_gates_per_timestep']})",
            f"  circuit evals / sample: {r['circuit_evaluations_per_sample']}",
            f"  params (classical/q)  : {r['classical_params']} / {r['quantum_params']}",
            f"  readout               : {self.cfg.readout} -> {self.cfg.n_classes} classes",
            f"  noisy device          : {r['noisy']}   diff: {r['diff_method']}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def build_model(cfg: Config, n_features: int, classical_lif: bool = False,
                stochastic: bool = False) -> QSNN:
    return QSNN(
        n_features=n_features,
        model_cfg=cfg.model,
        encoding_cfg=cfg.encoding,
        noise_cfg=cfg.noise,
        classical_lif=classical_lif,
        stochastic=stochastic,
    )
