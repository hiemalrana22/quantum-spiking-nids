"""Configuration objects for the QSNN-NIDS pipeline.

Everything the pipeline needs is expressed as a nested dataclass so that a run is
fully reproducible from a single YAML file (see ``configs/``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    # Registry key: unsw_nb15 | cicids2018 | cicids2017 | fiveg_nidd | edge_iiotset | synthetic
    dataset: str = "unsw_nb15"
    root: str = "data/raw"
    cache_dir: str = "data/processed"

    # Sub-sampling (these datasets are far too large for NISQ-scale experiments)
    max_rows: Optional[int] = 200_000
    balance: str = "undersample"          # undersample | none | class_weight
    binary: bool = True                   # benign vs attack; False -> multiclass
    min_class_count: int = 200            # multiclass: drop rarer classes

    # Feature pipeline
    scaler: str = "quantile"              # quantile | robust | standard | minmax
    select_k: int = 32                    # mutual-information pre-filter (0 = off)
    reducer: str = "pca"                  # pca | autoencoder | none
    n_features: int = 8                   # == n_qubits for angle/spike encoding

    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42
    n_jobs: int = -1


@dataclass
class EncodingConfig:
    """Feature vector -> spike train."""
    scheme: str = "phase"                 # phase | rate | latency | delta | repeat
    timesteps: int = 12                   # T
    gain: float = 1.0                     # scales firing probability / rate
    tau: float = 3.0                      # latency coding time-constant
    deterministic: bool = False           # rate coding without Bernoulli sampling
    seed: int = 42


@dataclass
class NoiseConfig:
    enabled: bool = False
    backend: str = "analytic"             # analytic | qiskit_aer | ibm_runtime
    depolarizing: float = 0.01            # per-qubit, per-block
    amplitude_damping: float = 0.005
    phase_damping: float = 0.0
    readout_error: float = 0.0            # bit-flip applied to measurement stats
    shots: Optional[int] = None           # None -> analytic expectation values
    ibm_backend: str = "ibm_brisbane"


@dataclass
class ModelConfig:
    n_qubits: int = 8
    n_qlif_layers: int = 2                # number of stacked QLIF populations
    ansatz_layers: int = 2                # variational depth inside each QLIF block
    ansatz: str = "strong"                # strong | basic | ring
    beta: float = 0.90                    # membrane leak
    learn_beta: bool = True
    threshold: float = 0.55               # firing threshold on P(|1>)
    learn_threshold: bool = False
    reset: str = "subtract"               # subtract | zero | none
    angle_map: str = "sigmoid"            # sigmoid (monotone) | tanh (symmetric, ablation)
    angle_gain: float = 2.0
    angle_bias: float = -2.0
    surrogate: str = "fast_sigmoid"       # fast_sigmoid | atan | sigmoid
    surrogate_alpha: float = 5.0
    readout: str = "rate_mem_prob"        # rate | mem | prob | rate_mem | rate_prob | rate_mem_prob
    dropout: float = 0.1
    n_classes: int = 2
    diff_method: str = "backprop"         # backprop | parameter-shift | adjoint


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-3
    quantum_lr: float = 1e-2              # separate LR for variational circuit params
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"             # cosine | plateau | none
    loss: str = "focal"                   # ce | focal | weighted_ce
    focal_gamma: float = 2.0
    grad_clip: float = 1.0
    patience: int = 8                     # early stopping
    device: str = "auto"                  # auto | cpu | cuda
    num_workers: int = 0
    amp: bool = False                     # keep off: quantum sim is float64-sensitive
    log_every: int = 20
    out_dir: str = "artifacts"
    run_name: str = "qsnn"
    save_best: bool = True
    seed: int = 42


@dataclass
class BaselineConfig:
    enabled: List[str] = field(
        default_factory=lambda: ["rf", "svm", "mlp", "qsvc", "qcnn"]
    )
    qsvc_max_train: int = 500             # O(n^2) circuits: 500~0.6h, 1000~2.5h, 2000~10h
    qsvc_reps: int = 2
    qcnn_epochs: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    # ---------------------------------------------------------------- helpers
    def __post_init__(self) -> None:
        # n_features and n_qubits must agree: one qubit per reduced feature.
        if self.data.n_features != self.model.n_qubits:
            self.model.n_qubits = self.data.n_features
        if self.data.binary:
            self.model.n_classes = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    # ``from __future__ import annotations`` stringifies field types, so the
    # section -> dataclass mapping is declared explicitly rather than inferred.
    SECTIONS = {
        "data": DataConfig,
        "encoding": EncodingConfig,
        "model": ModelConfig,
        "noise": NoiseConfig,
        "train": TrainConfig,
        "baselines": BaselineConfig,
    }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        unknown_sections = set(raw) - set(cls.SECTIONS)
        if unknown_sections:
            raise KeyError(f"unknown config sections: {sorted(unknown_sections)}")
        kwargs: Dict[str, Any] = {}
        for name, klass in cls.SECTIONS.items():
            sub = raw.get(name, {}) or {}
            if not isinstance(sub, dict):
                raise TypeError(f"config section '{name}' must be a mapping")
            unknown = set(sub) - {x.name for x in dataclasses.fields(klass)}
            if unknown:
                raise KeyError(f"unknown keys in '{name}': {sorted(unknown)}")
            kwargs[name] = klass(**sub)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def override(self, dotted: Dict[str, Any]) -> "Config":
        """Apply CLI overrides like {"train.epochs": 5, "noise.enabled": True}."""
        for key, value in dotted.items():
            if value is None:
                continue
            section, _, attr = key.partition(".")
            if not attr or not hasattr(self, section):
                raise KeyError(f"bad override '{key}'")
            target = getattr(self, section)
            if not hasattr(target, attr):
                raise KeyError(f"bad override '{key}'")
            setattr(target, attr, value)
        self.__post_init__()
        return self
