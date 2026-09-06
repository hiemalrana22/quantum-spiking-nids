#!/usr/bin/env python3
"""Noise-robustness sweep: train once (ideal), evaluate under increasing
depolarizing + amplitude-damping noise.

This is the experiment that ~60% of the surveyed papers skip.

    python scripts/eval_noise.py --config configs/unsw_nb15.yaml \
        --checkpoint artifacts/qsnn_unsw/best.pt --levels 0 0.005 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset, load_bundle
from qsnn.engine import evaluate, make_loaders, resolve_device, set_seed
from qsnn.metrics import format_metrics
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--damping-ratio", type=float, default=0.5,
                    help="amplitude_damping = ratio * depolarizing")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--out", default="artifacts/noise_sweep.json")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config)
    set_seed(cfg.train.seed)
    bundle = load_bundle(a.bundle) if a.bundle else build_dataset(cfg)
    cfg.model.n_classes = bundle.n_classes
    loaders = make_loaders(bundle, cfg)
    device = resolve_device(cfg.train.device)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    rows = []
    print(banner("NOISE ROBUSTNESS SWEEP"))
    for p in a.levels:
        cfg.noise.enabled = p > 0
        cfg.noise.backend = "analytic"
        cfg.noise.depolarizing = p
        cfg.noise.amplitude_damping = p * a.damping_ratio
        cfg.noise.shots = a.shots

        model = build_model(cfg, bundle.n_features)
        model.load_state_dict(state, strict=True)
        model.to(device)

        m, *_ = evaluate(model, loaders["test"], device, bundle.n_classes)
        m["depolarizing"] = p
        m["amplitude_damping"] = p * a.damping_ratio
        rows.append(m)
        print(f"  p={p:<6.3f}  {format_metrics(m, ['accuracy', 'f1', 'fpr', 'roc_auc'])}")

    base = rows[0]["f1"] if rows else 1.0
    print("\n  relative F1 retention vs ideal:")
    for r in rows:
        print(f"    p={r['depolarizing']:<6.3f}  {r['f1'] / max(base, 1e-9) * 100:6.2f}%")

    save_json({"config": cfg.to_dict(), "sweep": rows}, a.out)


if __name__ == "__main__":
    main()
