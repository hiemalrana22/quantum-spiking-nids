"""Training / evaluation loops for the QSNN.

Kept deliberately plain: the expensive part is the quantum simulation, so the
loop does nothing clever (no AMP, no compile) that would break autograd through
PennyLane.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data.preprocess import DataBundle
from .metrics import (confusion_table, detection_metrics, efficiency_metrics,
                      format_metrics, report)


# --------------------------------------------------------------------------- #
def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(bundle: DataBundle, cfg: Config) -> Dict[str, DataLoader]:
    def ds(X, y):
        return TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                             torch.as_tensor(y, dtype=torch.long))
    kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
              pin_memory=False)
    # drop_last on train only: the readout BatchNorm cannot handle a trailing
    # batch of size 1. Eval uses running statistics, so it is safe there.
    drop_last = len(bundle.X_train) % cfg.train.batch_size == 1
    return {
        "train": DataLoader(ds(bundle.X_train, bundle.y_train), shuffle=True,
                            drop_last=drop_last, **kw),
        "val": DataLoader(ds(bundle.X_val, bundle.y_val), shuffle=False,
                          drop_last=False, **kw),
        "test": DataLoader(ds(bundle.X_test, bundle.y_test), shuffle=False,
                           drop_last=False, **kw),
    }


# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Down-weights the easy benign majority - directly targets the
    class-imbalance gap flagged across the surveyed papers."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.numel() else None
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if w is not None:
            loss = loss * w.gather(0, target)
        return loss.mean()


def make_loss(cfg: Config, class_weights: np.ndarray, device: torch.device) -> nn.Module:
    w = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    if cfg.train.loss == "focal":
        return FocalLoss(cfg.train.focal_gamma, w).to(device)
    if cfg.train.loss == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Variational circuit params get their own (larger) learning rate: their
    loss landscape is much flatter than the classical layers'."""
    q, c = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (q if ("qweights" in name or "beta_logit" in name or "thr_logit" in name)
         else c).append(p)
    groups = [
        {"params": c, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
        {"params": q, "lr": cfg.train.quantum_lr, "weight_decay": 0.0},
    ]
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(groups)
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer '{cfg.train.optimizer}'")


def make_scheduler(opt, cfg: Config):
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                          factor=0.5, patience=3)
    return None


# --------------------------------------------------------------------------- #
@dataclass
class History:
    rows: List[Dict[str, float]] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rows.append(kw)

    def best(self, key: str = "val_f1") -> Dict[str, float]:
        return max(self.rows, key=lambda r: r.get(key, -1.0)) if self.rows else {}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.rows, fh, indent=2)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             n_classes: int = 2, criterion: Optional[nn.Module] = None
             ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, trues, losses = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            losses.append(float(criterion(logits, yb)) * len(yb))
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    y_score = np.concatenate(probs)
    y_true = np.concatenate(trues)
    y_pred = y_score.argmax(1)
    m = detection_metrics(y_true, y_pred, y_score, n_classes)
    if losses:
        m["loss"] = float(sum(losses) / max(len(y_true), 1))
    return m, y_true, y_pred, y_score


def train(model: nn.Module, loaders: Dict[str, DataLoader], cfg: Config,
          class_weights: np.ndarray, n_classes: int = 2,
          verbose: bool = True) -> Tuple[nn.Module, History]:
    device = resolve_device(cfg.train.device)
    model.to(device)
    criterion = make_loss(cfg, class_weights, device)
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    hist = History()

    best_f1, best_state, bad_epochs = -1.0, None, 0
    out_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, running, seen = time.perf_counter(), 0.0, 0
        for step, (xb, yb) in enumerate(loaders["train"], 1):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(loss.detach()) * len(yb)
            seen += len(yb)
            if verbose and cfg.train.log_every and step % cfg.train.log_every == 0:
                print(f"    ep{epoch:>3} step {step:>4}  loss {running / seen:.4f}")

        val, *_ = evaluate(model, loaders["val"], device, n_classes, criterion)
        if sched is not None:
            sched.step(val["f1"]) if cfg.train.scheduler == "plateau" else sched.step()

        row = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "secs": time.perf_counter() - t0,
               **{f"val_{k}": v for k, v in val.items() if isinstance(v, float)}}
        if hasattr(model, "spike_rate"):
            row["spike_rate"] = model.spike_rate
        hist.add(**row)
        if verbose:
            print(f"  epoch {epoch:>3}/{cfg.train.epochs}  "
                  f"loss {row['train_loss']:.4f}  "
                  f"val {format_metrics(val, ['accuracy', 'f1', 'fpr', 'roc_auc'])}  "
                  f"({row['secs']:.1f}s)")

        if val["f1"] > best_f1 + 1e-5:
            best_f1, bad_epochs = val["f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if cfg.train.save_best:
                torch.save({"state_dict": best_state, "config": cfg.to_dict(),
                            "val_f1": best_f1, "epoch": epoch},
                           os.path.join(out_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val f1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist.save(os.path.join(out_dir, "history.json"))
    return model, hist


# --------------------------------------------------------------------------- #
def full_evaluation(model, loaders, cfg: Config, bundle: DataBundle,
                    verbose: bool = True) -> Dict:
    device = resolve_device(cfg.train.device)
    n_classes = bundle.n_classes
    m, y_true, y_pred, y_score = evaluate(model, loaders["test"], device, n_classes)

    eff: Dict[str, float] = {}
    if hasattr(model, "resource_report"):
        xb = torch.as_tensor(bundle.X_test[: min(256, len(bundle.X_test))],
                             dtype=torch.float32).to(device)
        lat = model.benchmark_latency(xb)
        eff = efficiency_metrics(model.resource_report(), lat, model.spike_rate)

    out = {"detection": m, "efficiency": eff,
           "classes": list(bundle.class_names), "dataset": bundle.meta.get("dataset")}
    if verbose:
        print("\n=== TEST ===")
        print(" ", format_metrics(m))
        print("\n" + report(y_true, y_pred, bundle.class_names))
        print(confusion_table(y_true, y_pred, bundle.class_names))
        if eff:
            print("\n--- quantum resource footprint ---")
            for k, v in eff.items():
                print(f"  {k:<32} {v:.4f}" if isinstance(v, float) else f"  {k:<32} {v}")
    return out
