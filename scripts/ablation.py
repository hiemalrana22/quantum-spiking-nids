#!/usr/bin/env python3
"""Ablations reviewers will ask for: spike encoding scheme, qubit count,
timesteps, and quantum-vs-classical LIF.

    python scripts/ablation.py --config configs/unsw_nb15.yaml --study encoding
    python scripts/ablation.py --config configs/unsw_nb15.yaml --study qubits --epochs 10
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset, load_bundle
from qsnn.engine import evaluate, make_loaders, resolve_device, set_seed, train
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, results_table, save_json

STUDIES = {
    "encoding": ("encoding.scheme", ["rate", "latency", "phase", "delta", "repeat"]),
    "qubits": ("data.n_features", [4, 6, 8, 10, 12]),
    "timesteps": ("encoding.timesteps", [4, 8, 12, 16, 24]),
    "depth": ("model.ansatz_layers", [1, 2, 3, 4]),
    "ansatz": ("model.ansatz", ["ring", "basic", "strong"]),
    "beta": ("model.beta", [0.5, 0.7, 0.9, 0.95]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--study", required=True, choices=list(STUDIES) + ["quantum_vs_classical"])
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = Config.from_yaml(a.config).override({"train.epochs": a.epochs})
    rows = []

    if a.study == "quantum_vs_classical":
        variants = [("QSNN (quantum LIF)", False), ("SNN (classical LIF)", True)]
        for name, classical in variants:
            cfg = copy.deepcopy(base)
            cfg.train.run_name = f"abl_{name.split()[0].lower()}"
            set_seed(cfg.train.seed)
            bundle = load_bundle(a.bundle) if a.bundle else build_dataset(cfg, verbose=False)
            cfg.model.n_classes = bundle.n_classes
            model = build_model(cfg, bundle.n_features, classical_lif=classical)
            loaders = make_loaders(bundle, cfg)
            print(banner(name, "-"))
            model, _ = train(model, loaders, cfg, bundle.class_weights, bundle.n_classes)
            m, *_ = evaluate(model, loaders["test"], resolve_device(cfg.train.device),
                             bundle.n_classes)
            rows.append({"model": name, "family": "ablation", "metrics": m})
    else:
        key, values = STUDIES[a.study]
        for v in values:
            cfg = copy.deepcopy(base).override({key: v})
            cfg.train.run_name = f"abl_{a.study}_{v}"
            set_seed(cfg.train.seed)
            # qubit sweeps change the feature width -> rebuild the bundle
            rebuild = key == "data.n_features" or a.bundle is None
            bundle = (build_dataset(cfg, verbose=False) if rebuild
                      else load_bundle(a.bundle))
            cfg.model.n_classes = bundle.n_classes
            model = build_model(cfg, bundle.n_features)
            loaders = make_loaders(bundle, cfg)
            print(banner(f"{a.study} = {v}", "-"))
            model, _ = train(model, loaders, cfg, bundle.class_weights, bundle.n_classes)
            m, *_ = evaluate(model, loaders["test"], resolve_device(cfg.train.device),
                             bundle.n_classes)
            m.update({k: float(x) for k, x in model.resource_report().items()
                      if isinstance(x, (int, float))})
            rows.append({"model": f"{a.study}={v}", "family": "ablation", "metrics": m})

    print(banner(f"ABLATION: {a.study}"))
    print(results_table(rows))
    save_json(rows, a.out or f"artifacts/ablation_{a.study}.json")


if __name__ == "__main__":
    main()
