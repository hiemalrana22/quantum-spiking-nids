"""Quantum device factory: ideal simulator, noisy simulator, or IBM hardware.

Addresses the single most common gap in the surveyed literature - most QML-IDS
papers report simulator-only, noise-free numbers.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import pennylane as qml

from .config import NoiseConfig


def make_device(n_wires: int, noise: Optional[NoiseConfig] = None):
    """Return (device, noise_op, diff_ok).

    ``noise_op(wires)`` is a callable that injects the channel after each
    variational block; it is a no-op on ideal devices.
    """
    if noise is None or not noise.enabled:
        dev = qml.device("default.qubit", wires=n_wires, shots=None)
        return dev, (lambda wires: None), True

    if noise.backend == "analytic":
        # Density-matrix simulation: exact channels, still backprop-differentiable.
        dev = qml.device("default.mixed", wires=n_wires, shots=noise.shots)

        def noise_op(wires: List[int]) -> None:
            for w in wires:
                if noise.depolarizing > 0:
                    qml.DepolarizingChannel(noise.depolarizing, wires=w)
                if noise.amplitude_damping > 0:
                    qml.AmplitudeDamping(noise.amplitude_damping, wires=w)
                if noise.phase_damping > 0:
                    qml.PhaseDamping(noise.phase_damping, wires=w)

        return dev, noise_op, True

    if noise.backend == "qiskit_aer":
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install pennylane-qiskit qiskit-aer  (for backend='qiskit_aer')"
            ) from exc

        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(noise.depolarizing, 1),
                                       ["rx", "ry", "rz", "u1", "u2", "u3"])
        nm.add_all_qubit_quantum_error(depolarizing_error(noise.depolarizing * 5, 2),
                                       ["cx", "cz"])
        if noise.readout_error > 0:
            from qiskit_aer.noise import ReadoutError
            e = noise.readout_error
            nm.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))

        dev = qml.device(
            "qiskit.aer", wires=n_wires, shots=noise.shots or 4096, noise_model=nm,
        )
        # sampled device -> no backprop; caller must use parameter-shift
        return dev, (lambda wires: None), False

    if noise.backend == "ibm_runtime":  # pragma: no cover - needs credentials
        dev = qml.device(
            "qiskit.remote", wires=n_wires, backend=noise.ibm_backend,
            shots=noise.shots or 4096,
        )
        return dev, (lambda wires: None), False

    raise ValueError(f"unknown noise backend '{noise.backend}'")


def resolve_diff_method(requested: str, backprop_ok: bool) -> str:
    """Sampled/hardware devices cannot backprop; fall back to parameter-shift."""
    if backprop_ok:
        return requested
    if requested in ("backprop", "adjoint"):
        return "parameter-shift"
    return requested
