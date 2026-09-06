"""Dataset registry for QSNN-NIDS.

Each entry describes how to find the CSVs on disk, which columns leak identity
information (and must be dropped), and how to derive the binary / multiclass
label. Loading is deliberately tolerant: column names in the public releases of
these datasets are inconsistent (stray whitespace, case, unicode dashes), so we
normalise before matching.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Column-name normalisation
# --------------------------------------------------------------------------- #
def norm_col(name: str) -> str:
    """'  Destination Port ' -> 'destination_port'."""
    s = str(name).strip().lower()
    s = s.replace("﻿", "")
    s = re.sub(r"[\s\-/\.]+", "_", s)
    s = re.sub(r"[^0-9a-z_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class DatasetSpec:
    key: str
    name: str
    url: str
    glob_patterns: List[str]
    label_col: str                      # normalised name of the fine-grained label
    binary_label_col: Optional[str] = None   # normalised name of a ready 0/1 column
    benign_tokens: Sequence[str] = ("benign", "normal", "0")
    drop_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    read_kwargs: Dict = field(default_factory=dict)
    notes: str = ""


REGISTRY: Dict[str, DatasetSpec] = {
    # ------------------------------------------------------- PRIMARY DATASET
    "unsw_nb15": DatasetSpec(
        key="unsw_nb15",
        name="UNSW-NB15",
        url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        glob_patterns=[
            "**/UNSW_NB15_training-set.csv",
            "**/UNSW_NB15_testing-set.csv",
            "**/*training-set*.csv",
            "**/*UNSW*NB15*.csv",
            "**/*.csv",
        ],
        label_col="attack_cat",
        binary_label_col="label",
        benign_tokens=("normal",),
        drop_cols=["id", "srcip", "dstip", "sport", "dsport", "stime", "ltime"],
        categorical_cols=["proto", "service", "state"],
        notes=(
            "42 flow features, 9 attack families (Fuzzers, Analysis, Backdoors, "
            "DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms). Used by "
            "8 of the 25 surveyed papers -> direct comparability with the "
            "literature-review table (QSVC 99.78%, VQNN-QCNN 94.51%, QMGOA 99.89%). "
            "Use the pre-split UNSW_NB15_training-set.csv / testing-set.csv."
        ),
    ),
    # ----------------------------------------- SECOND / CROSS-DATASET BENCHMARK
    "cicids2017": DatasetSpec(
        key="cicids2017",
        name="CIC-IDS2017",
        url="https://www.unb.ca/cic/datasets/ids-2017.html",
        glob_patterns=[
            "**/*pcap_ISCX.csv",
            "**/MachineLearningCVE/*.csv",
            "**/*.csv",
        ],
        label_col="label",
        benign_tokens=("benign",),
        drop_cols=[
            "flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip",
            "source_port", "src_port", "timestamp", "external_ip",
            "fwd_header_length1", "unnamed_0",
        ],
        categorical_cols=["protocol"],
        notes=(
            "5 capture days, ~2.8M CICFlowMeter flows, 78 numeric features, 14 "
            "attack labels (DoS/DDoS, PortScan, Web Attack, Infiltration, Bot, "
            "Heartbleed, FTP/SSH-Patator). Timestamped, so it also supports the "
            "sequential 'window' spike-encoding mode. Comparability point: the "
            "hybrid QLSTM paper (97.43%) in the review. Use the "
            "MachineLearningCVE/ CSV folder."
        ),
    ),
}


def get_spec(key: str) -> DatasetSpec:
    if key not in REGISTRY:
        raise KeyError(
            f"unknown dataset '{key}'. Available: {sorted(REGISTRY)} (+ 'synthetic')"
        )
    return REGISTRY[key]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _find_files(root: str, patterns: Sequence[str]) -> List[str]:
    found: List[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        found.extend(h for h in hits if h.lower().endswith(".csv"))
        if found:
            break
    # de-duplicate, keep order
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_raw(
    dataset: str,
    root: str,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, DatasetSpec]:
    """Read (a sample of) a dataset from disk and normalise its column names.

    Rows are sampled per-file so that a capped run still sees every capture day
    / attack family rather than only the first CSV on disk.
    """
    spec = get_spec(dataset)
    files = _find_files(root, spec.glob_patterns)
    if not files:
        raise FileNotFoundError(
            f"No CSVs for '{spec.name}' under '{root}'.\n"
            f"Download it from: {spec.url}\n"
            f"Expected patterns: {spec.glob_patterns}"
        )

    per_file = None if max_rows is None else max(1, int(max_rows / len(files)) * 3)
    rng = np.random.default_rng(seed)
    frames: List[pd.DataFrame] = []

    for path in files:
        try:
            df = pd.read_csv(path, **spec.read_kwargs)
        except Exception as exc:                      # pragma: no cover
            print(f"[data]  skipping {os.path.basename(path)}: {exc}")
            continue
        if df.empty:
            continue
        df.columns = [norm_col(c) for c in df.columns]
        if per_file is not None and len(df) > per_file:
            idx = rng.choice(len(df), size=per_file, replace=False)
            df = df.iloc[np.sort(idx)]
        frames.append(df)
        print(f"[data]  {os.path.basename(path):<48} {len(df):>9,} rows")

    if not frames:
        raise RuntimeError(f"all CSVs under '{root}' failed to parse")

    # Align on the intersection of columns (CIC releases differ between days).
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common_cols = [c for c in frames[0].columns if c in common]
    df = pd.concat([f[common_cols] for f in frames], ignore_index=True)

    if max_rows is not None and len(df) > max_rows:
        idx = rng.choice(len(df), size=max_rows, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)

    print(f"[data]  combined: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df, spec


def make_synthetic(
    n_rows: int = 4000,
    n_features: int = 40,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, DatasetSpec]:
    """Offline stand-in with NIDS-like statistics (heavy tails, imbalance,
    a couple of categorical columns). Used by the smoke test and by CI."""
    from sklearn.datasets import make_classification

    weights = [0.8] + [0.2 / (n_classes - 1)] * (n_classes - 1) if n_classes > 1 else None
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=max(4, n_features // 3),
        n_redundant=max(2, n_features // 6),
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.02,
        class_sep=1.1,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    X[:, : n_features // 4] = np.exp(X[:, : n_features // 4])        # heavy tails
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["proto"] = rng.choice(["tcp", "udp", "icmp"], size=n_rows, p=[0.7, 0.25, 0.05])
    df["state"] = rng.choice(["con", "fin", "int", "req"], size=n_rows)
    names = ["benign"] + [f"attack_{i}" for i in range(1, max(2, n_classes))]
    df["attack_cat"] = [names[i] for i in y]
    df["label"] = (y != 0).astype(int)
    # sprinkle NaN/inf the way CICFlowMeter output does
    df.iloc[rng.choice(n_rows, 40, replace=False), 0] = np.inf
    df.iloc[rng.choice(n_rows, 40, replace=False), 1] = np.nan

    spec = DatasetSpec(
        key="synthetic", name="Synthetic-NIDS", url="(generated in-memory)",
        glob_patterns=[], label_col="attack_cat", binary_label_col="label",
        benign_tokens=("benign",), drop_cols=[],
        categorical_cols=["proto", "state"],
        notes="Not a real benchmark - offline smoke-test fixture only.",
    )
    return df, spec
