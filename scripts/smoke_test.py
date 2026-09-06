#!/usr/bin/env python3
"""End-to-end self-check on synthetic data - no dataset download needed.

Verifies: preprocessing -> spike encoding -> QLIF forward -> gradients reach
the variational circuit parameters -> training loop -> metrics -> noisy device
-> quantum baselines. Run this first to confirm the environment is sane, then
point train.py at the real CSVs.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from qsnn.config import Config
from qsnn.data.preprocess import build_dataset
from qsnn.data.spike_encoding import SCHEMES, SpikeEncoder, make_flow_windows
from qsnn.engine import evaluate, full_evaluation, make_loaders, set_seed, train
from qsnn.models.baselines import run_baselines
from qsnn.models.qsnn import build_model
from qsnn.utils import banner, results_table

CFG = "configs/smoke.yaml"
PASS, FAIL = "  [PASS]", "  [FAIL]"
results = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n>>> {name}")
            try:
                out = fn(*a, **k)
                print(f"{PASS} {name}")
                results.append((name, True, ""))
                return out
            except Exception as exc:
                print(f"{FAIL} {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                results.append((name, False, str(exc)))
                return None
        return wrapped
    return deco


@check("config loads + n_qubits syncs with n_features")
def t_config():
    cfg = Config.from_yaml(CFG)
    assert cfg.model.n_qubits == cfg.data.n_features
    cfg2 = Config.from_yaml(CFG).override({"data.n_features": 6})
    assert cfg2.model.n_qubits == 6, "override must resync n_qubits"
    return cfg


@check("preprocessing produces [0,1] features of the right width")
def t_data(cfg):
    b = build_dataset(cfg, verbose=False)
    assert b.X_train.shape[1] == cfg.data.n_features
    for X in (b.X_train, b.X_val, b.X_test):
        assert np.isfinite(X).all(), "non-finite values survived cleaning"
        assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6, "features outside [0,1]"
    assert len(b.class_names) == 2
    print(f"    train {b.X_train.shape}  val {b.X_val.shape}  test {b.X_test.shape}")
    return b


@check("every spike encoder emits binary [B,T,F] tensors")
def t_encoders(cfg, b):
    x = torch.as_tensor(b.X_train[:16])
    for scheme in SCHEMES:
        c = Config.from_yaml(CFG).override({"encoding.scheme": scheme}).encoding
        enc = SpikeEncoder(c)
        s = enc(x)
        assert s.shape == (16, c.timesteps, b.n_features), f"{scheme}: {s.shape}"
        if scheme != "repeat":
            uniq = set(torch.unique(s).tolist())
            assert uniq <= {0.0, 1.0}, f"{scheme} not binary: {uniq}"
        print(f"    {scheme:<9} shape {tuple(s.shape)}  sparsity {float(s.mean()):.3f}")


@check("flow windowing (sequence mode for CIC-IDS2017)")
def t_windows(b):
    Xw, yw = make_flow_windows(b.X_test, b.y_test, T=5, stride=2)
    assert Xw.ndim == 3 and Xw.shape[1] == 5 and len(Xw) == len(yw)
    print(f"    windows {Xw.shape}  attack fraction {yw.mean():.3f}")


@check("QSNN forward pass + resource report")
def t_forward(cfg, b):
    model = build_model(cfg, b.n_features)
    print(model.describe())
    x = torch.as_tensor(b.X_test[:8])
    logits = model(x)
    assert logits.shape == (8, b.n_classes), logits.shape
    assert torch.isfinite(logits).all()
    r = model.resource_report()
    assert r["depth_per_timestep"] > 0, "circuit specs failed"
    assert r["quantum_params"] > 0
    print(f"    depth/step {r['depth_per_timestep']}  "
          f"evals/sample {r['circuit_evaluations_per_sample']}  "
          f"qparams {r['quantum_params']}")
    return model


@check("gradients reach the variational circuit parameters")
def t_grads(model, b):
    x = torch.as_tensor(b.X_train[:8])
    y = torch.as_tensor(b.y_train[:8], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()
    named = dict(model.named_parameters())
    qg = [(n, p) for n, p in named.items() if "qweights" in n]
    assert qg, "no quantum parameters found"
    for n, p in qg:
        assert p.grad is not None, f"{n} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{n} gradient not finite"
        assert p.grad.abs().sum() > 0, f"{n} gradient is identically zero"
        print(f"    {n}: |grad| = {float(p.grad.abs().mean()):.3e}")
    for n in ("qlif_layers.0.beta_logit",):
        if n in named and named[n].grad is not None:
            print(f"    {n}: |grad| = {float(named[n].grad.abs().mean()):.3e}")


@check("membrane dynamics: leak, reset and monotone drive response")
def t_dynamics(cfg, b):
    from qsnn.models.qlif import QuantumLIFLayer
    layer = QuantumLIFLayer(n_qubits=cfg.model.n_qubits, ansatz_layers=1,
                            ansatz="ring", beta=0.9, threshold=0.5)
    T, B = 8, 4
    strong = torch.ones(B, T, cfg.model.n_qubits) * 1.5
    weak = torch.ones(B, T, cfg.model.n_qubits) * 0.02
    s_hi, st_hi = layer(strong, return_states=True)
    s_lo, _ = layer(weak, return_states=True)
    print(f"    spike rate  strong-drive {float(s_hi.mean()):.3f}   "
          f"weak-drive {float(s_lo.mean()):.3f}")
    assert float(s_hi.mean()) >= float(s_lo.mean()), "firing rate must rise with drive"
    p = st_hi["p_excited"]
    assert ((p >= 0) & (p <= 1)).all(), "P(|1>) outside [0,1]"
    # no input -> membrane must decay towards 0
    zero = torch.zeros(B, T, cfg.model.n_qubits)
    _, st0 = layer(zero, return_states=True)
    assert float(st0["mem"].abs().max()) < 1e-5, "membrane must stay at rest with no input"


@check("training loop runs and metrics are well-formed")
def t_train(cfg, b, model):
    loaders = make_loaders(b, cfg)
    model, hist = train(model, loaders, cfg, b.class_weights, b.n_classes, verbose=True)
    assert len(hist.rows) >= 1
    res = full_evaluation(model, loaders, cfg, b, verbose=False)
    d = res["detection"]
    for k in ("accuracy", "precision", "recall", "f1", "fpr"):
        assert k in d and 0.0 <= d[k] <= 1.0, f"bad metric {k}={d.get(k)}"
    e = res["efficiency"]
    assert e["n_qubits"] == cfg.model.n_qubits
    assert e["latency_ms_per_sample"] > 0
    print(f"    test acc {d['accuracy']:.4f}  f1 {d['f1']:.4f}  fpr {d['fpr']:.4f}")
    print(f"    latency {e['latency_ms_per_sample']:.2f} ms/flow  "
          f"spike rate {e['spike_rate']:.3f}")
    return model, loaders, res


@check("noisy device (depolarizing + amplitude damping) runs and stays differentiable")
def t_noise(b):
    cfg = Config.from_yaml(CFG).override({"noise.enabled": True})
    model = build_model(cfg, b.n_features)
    x = torch.as_tensor(b.X_test[:4])
    y = torch.as_tensor(b.y_test[:4], dtype=torch.long)
    logits = model(x)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, y).backward()
    g = dict(model.named_parameters())["qlif_layers.0.qweights"].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"    noisy forward OK, |grad| = {float(g.abs().mean()):.3e}")


@check("classical-LIF ablation runs")
def t_ablation(cfg, b):
    model = build_model(cfg, b.n_features, classical_lif=True)
    out = model(torch.as_tensor(b.X_test[:8]))
    assert out.shape == (8, b.n_classes)
    print(f"    classical-LIF logits {tuple(out.shape)}")


@check("baselines (classical + quantum) run on the same bundle")
def t_baselines(cfg, b, qsnn_res):
    rows = [{"model": "QSNN (ours)", "family": "quantum", "metrics": qsnn_res["detection"]}]
    rows += run_baselines(b, cfg, ["rf", "mlp", "qsvc", "qcnn"], verbose=False)
    assert len(rows) >= 3, "baselines did not produce results"
    print(results_table(rows))


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(banner("QSNN-NIDS smoke test (synthetic data, no download)"))
    set_seed(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    cfg = t_config()
    b = t_data(cfg)
    t_encoders(cfg, b)
    t_windows(b)
    model = t_forward(cfg, b)
    t_grads(model, b)
    t_dynamics(cfg, b)
    trained = t_train(cfg, b, model)
    t_noise(b)
    t_ablation(cfg, b)
    if trained:
        t_baselines(cfg, b, trained[2])

    print(banner("SMOKE TEST SUMMARY"))
    ok = sum(1 for _, p, _ in results if p)
    for name, passed, err in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  <- {err}" if err else ""))
    print(f"\n  {ok}/{len(results)} checks passed")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
