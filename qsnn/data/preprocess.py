"""Cleaning -> encoding -> scaling -> selection -> dimensionality reduction.

The output of :func:`build_dataset` is a bundle of arrays in ``[0, 1]`` with
exactly ``cfg.data.n_features`` columns, which is what the spike encoder and the
quantum layers expect (one qubit per feature).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ..config import Config
from .datasets import DatasetSpec, load_raw, make_synthetic, norm_col


# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    class_weights: np.ndarray
    meta: Dict

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        rows = [
            f"dataset      : {self.meta.get('dataset')}",
            f"features     : {self.n_features}  {self.feature_names}",
            f"classes      : {self.n_classes}  {self.class_names}",
            f"train/val/test: {len(self.y_train):,} / {len(self.y_val):,} / {len(self.y_test):,}",
            f"train balance: {np.bincount(self.y_train, minlength=self.n_classes).tolist()}",
            f"class weights: {np.round(self.class_weights, 3).tolist()}",
        ]
        return "\n".join("  " + r for r in rows)


# --------------------------------------------------------------------------- #
# Step 1 - cleaning
# --------------------------------------------------------------------------- #
def clean_frame(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]

    # Identity / leakage columns: IPs, ports and timestamps let a model memorise
    # the capture schedule instead of learning traffic behaviour.
    drop = [c for c in spec.drop_cols if c in df.columns]
    drop += [c for c in df.columns if c.startswith("unnamed")]
    if drop:
        df = df.drop(columns=list(dict.fromkeys(drop)))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    # constant columns carry no signal but do consume PCA components
    nunique = df.nunique(dropna=False)
    const = [c for c in df.columns if nunique.get(c, 2) <= 1]
    if const:
        df = df.drop(columns=const)
    df = df.drop_duplicates()
    return df


# --------------------------------------------------------------------------- #
# Step 2 - labels
# --------------------------------------------------------------------------- #
def extract_labels(
    df: pd.DataFrame, spec: DatasetSpec, binary: bool, min_class_count: int
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    label_cols = [c for c in (spec.label_col, spec.binary_label_col) if c and c in df.columns]
    if not label_cols:
        cands = [c for c in df.columns if c in ("label", "attack_cat", "attack_type",
                                                "attack_label", "class", "type")]
        if not cands:
            raise KeyError(f"no label column found; columns={list(df.columns)[:40]}")
        label_cols = cands

    fine_col = spec.label_col if spec.label_col in df.columns else label_cols[0]
    raw = df[fine_col]
    if pd.api.types.is_numeric_dtype(raw) and spec.binary_label_col in df.columns \
            and spec.binary_label_col != fine_col:
        raw = df[spec.binary_label_col]

    raw = raw.astype(str).str.strip().str.lower()
    raw = raw.str.replace(r"[\s\-]+", "_", regex=True)
    benign = {str(t).lower() for t in spec.benign_tokens}

    X = df.drop(columns=[c for c in set(label_cols) | {fine_col} if c in df.columns])

    if binary:
        y = (~raw.isin(benign)).astype(int).to_numpy()
        return X, y, ["benign", "attack"]

    counts = raw.value_counts()
    keep = counts[counts >= min_class_count].index.tolist()
    mask = raw.isin(keep).to_numpy()
    X, raw = X.loc[mask], raw.loc[mask]
    # benign first so index 0 is always the negative class
    classes = sorted(keep, key=lambda c: (c not in benign, c))
    mapping = {c: i for i, c in enumerate(classes)}
    y = raw.map(mapping).to_numpy().astype(int)
    return X, y, classes


# --------------------------------------------------------------------------- #
# Step 3 - feature matrix
# --------------------------------------------------------------------------- #
def encode_features(X: pd.DataFrame, spec: DatasetSpec) -> Tuple[np.ndarray, List[str]]:
    X = X.copy()
    cat = [c for c in X.columns
           if c in spec.categorical_cols or X[c].dtype == object
           or isinstance(X[c].dtype, pd.CategoricalDtype)]

    for c in cat:
        s = X[c].astype(str).str.strip().str.lower()
        top = s.value_counts().nlargest(15).index          # cap one-hot width
        X[c] = s.where(s.isin(top), "other")
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False, dtype=np.float32)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.to_numpy(dtype=np.float32), list(X.columns)


def make_scaler(kind: str, seed: int):
    if kind == "quantile":
        # robust to the heavy tails in byte/duration counters
        return QuantileTransformer(output_distribution="uniform",
                                   n_quantiles=1000, subsample=200_000,
                                   random_state=seed)
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"unknown scaler '{kind}'")


# --------------------------------------------------------------------------- #
# Step 4 - dimensionality reduction to n_qubits
# --------------------------------------------------------------------------- #
class _Autoencoder(nn.Module):
    def __init__(self, d_in: int, d_lat: int):
        super().__init__()
        h = max(d_lat * 4, min(128, d_in))
        self.enc = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, d_lat), nn.Tanh(),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_lat, h // 2), nn.ReLU(),
            nn.Linear(h // 2, h), nn.ReLU(),
            nn.Linear(h, d_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def fit_autoencoder(X: np.ndarray, d_lat: int, seed: int,
                    epochs: int = 40, batch_size: int = 256) -> _Autoencoder:
    torch.manual_seed(seed)
    model = _Autoencoder(X.shape[1], d_lat)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xb_all = torch.as_tensor(X, dtype=torch.float32)
    n = len(xb_all)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            xb = xb_all[perm[i:i + batch_size]]
            rec, _ = model(xb)
            loss = nn.functional.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"[ae]    epoch {ep + 1:>3}  recon-mse {tot / n:.5f}")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Step 5 - class balance
# --------------------------------------------------------------------------- #
def undersample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    target = int(counts[counts > 0].min())
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=target, replace=False)
        for c in np.flatnonzero(counts > 0)
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_dataset(cfg: Config, verbose: bool = True) -> DataBundle:
    d = cfg.data
    if d.dataset == "synthetic":
        df, spec = make_synthetic(
            n_rows=d.max_rows or 4000,
            n_classes=2 if d.binary else 4,
            seed=d.seed,
        )
    else:
        df, spec = load_raw(d.dataset, d.root, d.max_rows, d.seed)

    df = clean_frame(df, spec)
    X_df, y, class_names = extract_labels(df, spec, d.binary, d.min_class_count)
    X, feat_names = encode_features(X_df, spec)
    if verbose:
        print(f"[prep]  after encoding: {X.shape[0]:,} x {X.shape[1]} "
              f"({len(class_names)} classes)")

    # --- split first, fit every transform on train only (no leakage) --------
    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=d.test_size, random_state=d.seed, stratify=strat)
    strat = y_tr if np.bincount(y_tr).min() >= 2 else None
    val_frac = d.val_size / (1.0 - d.test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_frac, random_state=d.seed, stratify=strat)

    if d.balance == "undersample":
        X_tr, y_tr = undersample(X_tr, y_tr, d.seed)

    scaler = make_scaler(d.scaler, d.seed).fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a).astype(np.float32)
                        for a in (X_tr, X_va, X_te))

    # --- mutual-information pre-filter -------------------------------------
    if d.select_k and 0 < d.select_k < X_tr.shape[1]:
        sub = min(len(X_tr), 20_000)
        rng = np.random.default_rng(d.seed)
        s = rng.choice(len(X_tr), size=sub, replace=False)
        mi = mutual_info_classif(X_tr[s], y_tr[s], random_state=d.seed)
        keep = np.argsort(mi)[::-1][:d.select_k]
        keep = np.sort(keep)
        X_tr, X_va, X_te = X_tr[:, keep], X_va[:, keep], X_te[:, keep]
        feat_names = [feat_names[i] for i in keep]
        if verbose:
            print(f"[prep]  MI filter -> {len(keep)} features; top-5: "
                  f"{[feat_names[i] for i in np.argsort(mi[keep])[::-1][:5]]}")

    # --- reduce to n_qubits dimensions --------------------------------------
    reducer_info: Dict = {"kind": d.reducer}
    if d.reducer == "pca" and X_tr.shape[1] > d.n_features:
        pca = PCA(n_components=d.n_features, random_state=d.seed, whiten=False).fit(X_tr)
        X_tr, X_va, X_te = (pca.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te))
        feat_names = [f"pc{i + 1}" for i in range(d.n_features)]
        reducer_info["explained_variance"] = float(pca.explained_variance_ratio_.sum())
        if verbose:
            print(f"[prep]  PCA -> {d.n_features} dims "
                  f"(explained variance {reducer_info['explained_variance']:.3f})")
    elif d.reducer == "autoencoder" and X_tr.shape[1] > d.n_features:
        ae = fit_autoencoder(X_tr, d.n_features, d.seed)
        with torch.no_grad():
            X_tr, X_va, X_te = (
                ae.enc(torch.as_tensor(a, dtype=torch.float32)).numpy().astype(np.float32)
                for a in (X_tr, X_va, X_te))
        feat_names = [f"z{i + 1}" for i in range(d.n_features)]
    elif X_tr.shape[1] > d.n_features:
        X_tr, X_va, X_te = (a[:, :d.n_features] for a in (X_tr, X_va, X_te))
        feat_names = feat_names[:d.n_features]

    # --- final squash into [0, 1] for spike/angle encoding ------------------
    final = MinMaxScaler(feature_range=(0.0, 1.0)).fit(X_tr)
    X_tr, X_va, X_te = (np.clip(final.transform(a), 0.0, 1.0).astype(np.float32)
                        for a in (X_tr, X_va, X_te))

    bundle = DataBundle(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
        feature_names=feat_names, class_names=class_names,
        class_weights=compute_class_weights(y_tr, len(class_names)),
        meta={"dataset": spec.name, "dataset_key": spec.key, "source": spec.url,
              "reducer": reducer_info, "scaler": d.scaler, "binary": d.binary,
              "seed": d.seed},
    )
    if verbose:
        print("[prep]  bundle ready\n" + bundle.summary())
    return bundle


# --------------------------------------------------------------------------- #
def save_bundle(bundle: DataBundle, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X_train=bundle.X_train, y_train=bundle.y_train,
        X_val=bundle.X_val, y_val=bundle.y_val,
        X_test=bundle.X_test, y_test=bundle.y_test,
        feature_names=np.array(bundle.feature_names, dtype=object),
        class_names=np.array(bundle.class_names, dtype=object),
        class_weights=bundle.class_weights,
        meta=np.array([json.dumps(bundle.meta)], dtype=object),
    )
    print(f"[prep]  saved -> {path}")


def load_bundle(path: str) -> DataBundle:
    z = np.load(path, allow_pickle=True)
    return DataBundle(
        X_train=z["X_train"], y_train=z["y_train"],
        X_val=z["X_val"], y_val=z["y_val"],
        X_test=z["X_test"], y_test=z["y_test"],
        feature_names=list(z["feature_names"]),
        class_names=list(z["class_names"]),
        class_weights=z["class_weights"],
        meta=json.loads(z["meta"][0]),
    )
