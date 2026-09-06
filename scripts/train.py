#!/usr/bin/env python3
"""Train + evaluate the QSNN.

    python scripts/train.py --config configs/unsw_nb15.yaml
    python scripts/train.py --config configs/unsw_nb15.yaml --bundle data/processed/unsw_nb15.npz
    python scripts/train.py --config configs/cicids2017.yaml --noise --epochs 15
    python scripts/train.py --config configs/unsw_nb15.yaml --baselines rf svm mlp qsvc qcnn
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset, load_bundle
from qsnn.engine import full_evaluation, make_loaders, set_seed, train
from qsnn.models.baselines import run_baselines
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, results_table, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", default=None, help="cached .npz from prepare_data.py")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--qubits", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--encoding", default=None,
                    choices=["rate", "latency", "phase", "delta", "repeat"])
    ap.add_argument("--noise", action="store_true", help="train on a noisy device")
    ap.add_argument("--classical-lif", action="store_true",
                    help="ablation: swap the qubit for a classical LIF neuron")
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="e.g. --baselines rf svm mlp qsvc qcnn  (empty = skip)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config).override({
        "train.epochs": a.epochs, "train.batch_size": a.batch_size, "train.lr": a.lr,
        "data.n_features": a.qubits, "encoding.timesteps": a.timesteps,
        "encoding.scheme": a.encoding, "noise.enabled": True if a.noise else None,
        "train.run_name": a.run_name, "train.device": a.device,
        "train.seed": a.seed, "data.seed": a.seed,
    })
    set_seed(cfg.train.seed)

    print(banner(f"QSNN-NIDS  |  {cfg.data.dataset}  |  run '{cfg.train.run_name}'"))
    bundle = load_bundle(a.bundle) if a.bundle else build_dataset(cfg)
    cfg.model.n_classes = bundle.n_classes

    model = build_model(cfg, bundle.n_features, classical_lif=a.classical_lif)
    print(banner("MODEL", "-"))
    print(model.describe())

    loaders = make_loaders(bundle, cfg)
    print(banner("TRAINING", "-"))
    model, hist = train(model, loaders, cfg, bundle.class_weights, bundle.n_classes)

    res = full_evaluation(model, loaders, cfg, bundle)
    rows = [{"model": "QSNN (ours)", "family": "quantum", "metrics": res["detection"]}]

    which = a.baselines if a.baselines is not None else cfg.baselines.enabled
    if which:
        print(banner("BASELINES (same preprocessing, same split)", "-"))
        rows += run_baselines(bundle, cfg, which)

    print(banner("SUMMARY"))
    print(results_table(rows))

    out_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    save_json({"config": cfg.to_dict(), "data": bundle.meta,
               "qsnn": res, "comparison": rows,
               "history_best": hist.best()},
              os.path.join(out_dir, "results.json"))
    cfg.save(os.path.join(out_dir, "config.used.yaml"))
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(out_dir, "final.pt"))
    print(f"\nartifacts -> {out_dir}/")


if __name__ == "__main__":
    main()
