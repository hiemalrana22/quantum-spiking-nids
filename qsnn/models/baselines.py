"""Baselines run on the *identical* preprocessing pipeline.

Most of the surveyed papers compare against numbers quoted from other papers,
which were produced on different splits and different feature pipelines. Here
every baseline consumes the exact same DataBundle, so the comparison is fair.

Classical : Random Forest, RBF-SVM, MLP   (PPT objective 4)
Quantum   : QSVC (quantum kernel + SVM), QCNN (variational conv/pool)
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..config import Config
from ..metrics import detection_metrics


# --------------------------------------------------------------------------- #
# Classical
# --------------------------------------------------------------------------- #
def run_random_forest(bundle, cfg: Config) -> Dict:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=cfg.data.n_jobs,
        random_state=cfg.data.seed)
    t0 = time.perf_counter()
    clf.fit(bundle.X_train, bundle.y_train)
    fit_s = time.perf_counter() - t0
    p = clf.predict_proba(bundle.X_test)
    m = detection_metrics(bundle.y_test, p.argmax(1), p, bundle.n_classes)
    m["fit_seconds"] = fit_s
    return {"model": "RandomForest", "family": "classical", "metrics": m}


def run_svm(bundle, cfg: Config, max_train: int = 20_000) -> Dict:
    from sklearn.svm import SVC
    rng = np.random.default_rng(cfg.data.seed)
    n = min(max_train, len(bundle.X_train))
    idx = rng.choice(len(bundle.X_train), n, replace=False)
    clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced",
              probability=True, random_state=cfg.data.seed)
    t0 = time.perf_counter()
    clf.fit(bundle.X_train[idx], bundle.y_train[idx])
    fit_s = time.perf_counter() - t0
    p = clf.predict_proba(bundle.X_test)
    m = detection_metrics(bundle.y_test, p.argmax(1), p, bundle.n_classes)
    m["fit_seconds"] = fit_s
    return {"model": "RBF-SVM", "family": "classical", "metrics": m}


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _torch_fit(model, bundle, cfg: Config, epochs: int, lr: float = 1e-3,
               tag: str = "dnn", verbose: bool = True) -> Dict:
    from ..engine import resolve_device
    device = resolve_device(cfg.train.device)
    model.to(device)
    w = torch.as_tensor(bundle.class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xtr = torch.as_tensor(bundle.X_train, dtype=torch.float32)
    ytr = torch.as_tensor(bundle.y_train, dtype=torch.long)
    bs = cfg.train.batch_size
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            j = perm[i:i + bs]
            xb, yb = Xtr[j].to(device), ytr[j].to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(yb)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [{tag}] epoch {ep + 1}/{epochs}  loss {tot / len(Xtr):.4f}")
    fit_s = time.perf_counter() - t0

    model.eval()
    probs = []
    Xte = torch.as_tensor(bundle.X_test, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            probs.append(torch.softmax(model(Xte[i:i + bs].to(device)), -1).cpu().numpy())
    p = np.concatenate(probs)
    m = detection_metrics(bundle.y_test, p.argmax(1), p, bundle.n_classes)
    m["fit_seconds"] = fit_s
    return m


def run_mlp(bundle, cfg: Config, epochs: int = 40, verbose: bool = True) -> Dict:
    model = MLP(bundle.n_features, bundle.n_classes)
    m = _torch_fit(model, bundle, cfg, epochs, tag="mlp", verbose=verbose)
    m["params"] = sum(p.numel() for p in model.parameters())
    return {"model": "MLP(DNN)", "family": "classical", "metrics": m}


# --------------------------------------------------------------------------- #
# Quantum baseline 1 - QSVC (quantum kernel + classical SVM)
# --------------------------------------------------------------------------- #
def make_kernel_fn(n_qubits: int, reps: int = 2, noise_cfg=None):
    from ..noise import make_device
    dev, noise_op, _ = make_device(n_qubits, noise_cfg)
    wires = list(range(n_qubits))

    def feature_map(x):
        # ZZ-style second-order map: the encoding used by the QSVC/QuIDS papers.
        for r in range(reps):
            for i in wires:
                qml.Hadamard(wires=i)
                qml.RZ(np.pi * x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(np.pi * (1 - x[i]) * (1 - x[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            noise_op(wires)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=wires)

    def kernel(x1, x2) -> float:
        return float(kernel_circuit(x1, x2)[0])          # |<phi(x2)|phi(x1)>|^2

    return kernel


def run_qsvc(bundle, cfg: Config, verbose: bool = True) -> Dict:
    from sklearn.svm import SVC
    n_q = cfg.model.n_qubits
    max_n = cfg.baselines.qsvc_max_train
    rng = np.random.default_rng(cfg.data.seed)

    n_tr = min(max_n, len(bundle.X_train))
    n_te = min(max_n // 2, len(bundle.X_test))
    itr = rng.choice(len(bundle.X_train), n_tr, replace=False)
    ite = rng.choice(len(bundle.X_test), n_te, replace=False)
    Xtr, ytr = bundle.X_train[itr], bundle.y_train[itr]
    Xte, yte = bundle.X_test[ite], bundle.y_test[ite]

    kfn = make_kernel_fn(n_q, cfg.baselines.qsvc_reps, cfg.noise)
    if verbose:
        print(f"    [qsvc] kernel matrix {n_tr}x{n_tr} + {n_te}x{n_tr} "
              f"({n_tr * (n_tr + 1) // 2 + n_te * n_tr:,} circuits)")

    t0 = time.perf_counter()
    K_tr = qml.kernels.square_kernel_matrix(Xtr, kfn, assume_normalized_kernel=True)
    K_te = qml.kernels.kernel_matrix(Xte, Xtr, kfn)
    clf = SVC(kernel="precomputed", C=10.0, class_weight="balanced",
              probability=True, random_state=cfg.data.seed).fit(K_tr, ytr)
    fit_s = time.perf_counter() - t0

    p = clf.predict_proba(K_te)
    m = detection_metrics(yte, p.argmax(1), p, bundle.n_classes)
    m["fit_seconds"] = fit_s
    m["n_train_used"] = n_tr
    m["n_test_used"] = n_te
    return {"model": "QSVC (ZZ kernel)", "family": "quantum", "metrics": m}


# --------------------------------------------------------------------------- #
# Quantum baseline 2 - QCNN (variational convolution + pooling)
# --------------------------------------------------------------------------- #
class QCNN(nn.Module):
    """Cong-style QCNN: alternating two-qubit conv unitaries and pooling
    (measure-and-discard) until log2(n) qubits remain."""

    def __init__(self, n_qubits: int, n_classes: int = 2, noise_cfg=None,
                 diff_method: str = "backprop"):
        super().__init__()
        from ..noise import make_device, resolve_diff_method
        self.n_qubits = n_qubits
        dev, noise_op, ok = make_device(n_qubits, noise_cfg)
        diff = resolve_diff_method(diff_method, ok)

        # layer schedule: halve the register each block
        self.wire_sets, wires = [], list(range(n_qubits))
        while len(wires) > 1:
            self.wire_sets.append(list(wires))
            wires = wires[::2]
        self.out_wires = wires

        shapes = [(len(w) // 2, 15) for w in self.wire_sets]   # 15 = SU(4) params
        self.qparams = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(s)) for s in shapes])

        wire_sets = self.wire_sets
        out_wires = self.out_wires

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, *params):
            qml.AngleEmbedding(np.pi * inputs, wires=range(n_qubits), rotation="Y")
            noise_op(list(range(n_qubits)))
            for ws, prm in zip(wire_sets, params):
                for k in range(len(ws) // 2):
                    qml.ArbitraryUnitary(prm[k], wires=[ws[2 * k], ws[2 * k + 1]])
                noise_op(ws)
            return [qml.expval(qml.PauliZ(w)) for w in out_wires]

        self.circuit = circuit
        self.head = nn.Linear(len(out_wires), n_classes)

    def forward(self, x):
        out = self.circuit(x, *self.qparams)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=-1)
        out = out.to(x.dtype)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return self.head(out)


def run_qcnn(bundle, cfg: Config, verbose: bool = True) -> Dict:
    model = QCNN(cfg.model.n_qubits, bundle.n_classes, cfg.noise, cfg.model.diff_method)
    m = _torch_fit(model, bundle, cfg, cfg.baselines.qcnn_epochs, lr=5e-3,
                   tag="qcnn", verbose=verbose)
    m["params"] = sum(p.numel() for p in model.parameters())
    return {"model": "QCNN", "family": "quantum", "metrics": m}


# --------------------------------------------------------------------------- #
RUNNERS = {
    "rf": run_random_forest,
    "svm": run_svm,
    "mlp": run_mlp,
    "qsvc": run_qsvc,
    "qcnn": run_qcnn,
}


def run_baselines(bundle, cfg: Config, which=None, verbose: bool = True) -> list:
    which = which or cfg.baselines.enabled
    results = []
    for key in which:
        if key not in RUNNERS:
            print(f"[baseline] unknown '{key}', skipping")
            continue
        if verbose:
            print(f"\n[baseline] {key} ...")
        try:
            res = RUNNERS[key](bundle, cfg) if key in ("rf", "svm") \
                else RUNNERS[key](bundle, cfg, verbose=verbose)
            results.append(res)
            if verbose:
                mm = res["metrics"]
                print(f"    -> acc {mm['accuracy']:.4f}  f1 {mm['f1']:.4f}  "
                      f"fpr {mm.get('fpr', float('nan')):.4f}")
        except Exception as exc:                              # pragma: no cover
            print(f"[baseline] {key} FAILED: {type(exc).__name__}: {exc}")
    return results
