#!/usr/bin/env python3
"""Build and cache a preprocessed DataBundle.

    python scripts/prepare_data.py --config configs/unsw_nb15.yaml
    python scripts/prepare_data.py --config configs/cicids2017.yaml --max-rows 300000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset, save_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-features", type=int, default=None, help="== n_qubits")
    ap.add_argument("--multiclass", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config).override({
        "data.root": a.root,
        "data.max_rows": a.max_rows,
        "data.n_features": a.n_features,
        "data.binary": False if a.multiclass else None,
    })

    bundle = build_dataset(cfg)
    out = a.out or os.path.join(cfg.data.cache_dir, f"{cfg.data.dataset}.npz")
    save_bundle(bundle, out)


if __name__ == "__main__":
    main()
