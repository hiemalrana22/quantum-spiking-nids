"""Detection metrics + the efficiency metrics that differentiate this work.

The surveyed papers almost universally report accuracy only. For an IDS the
false-positive rate is what decides whether the thing is deployable (analyst
alert fatigue), and for a NISQ model the qubit/depth/latency budget is what
decides whether it is runnable - so both are first-class here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    avg = "binary" if n_classes == 2 else "macro"

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        out.update({
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            # false-positive rate: the metric that decides analyst workload
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "tnr_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        })
    else:
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tn = cm.sum() - (fp + fn + np.diag(cm))
        with np.errstate(divide="ignore", invalid="ignore"):
            out["fpr"] = float(np.nanmean(fp / np.maximum(fp + tn, 1)))
            out["fnr"] = float(np.nanmean(fn / np.maximum(fn + np.diag(cm), 1)))

    if y_score is not None and len(np.unique(y_true)) > 1:
        y_score = np.asarray(y_score)
        try:
            if n_classes == 2:
                s = y_score[:, 1] if y_score.ndim > 1 else y_score
                out["roc_auc"] = float(roc_auc_score(y_true, s))
                out["pr_auc"] = float(average_precision_score(y_true, s))
            else:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return out


def efficiency_metrics(resource_report: Dict, latency: Optional[Dict] = None,
                       spike_rate: Optional[float] = None) -> Dict[str, float]:
    """Quantum-resource footprint. This is the paper's differentiator: none of
    the 25 surveyed works reports depth + qubits + latency + sparsity together."""
    r = dict(resource_report)
    out = {
        "n_qubits": r.get("n_qubits", 0),
        "circuit_depth_per_timestep": r.get("depth_per_timestep", 0),
        "total_circuit_depth": r.get("total_circuit_depth", 0),
        "two_qubit_gates_per_timestep": r.get("two_qubit_gates_per_timestep", 0),
        "circuit_evals_per_sample": r.get("circuit_evaluations_per_sample", 0),
        "quantum_params": r.get("quantum_params", 0),
        "classical_params": r.get("classical_params", 0),
        "total_params": r.get("total_params", 0),
        "timesteps": r.get("timesteps", 0),
    }
    if latency:
        out.update({k: float(v) for k, v in latency.items()})
    if spike_rate is not None:
        out["spike_rate"] = float(spike_rate)
        # Spiking hardware only pays for events; 1 - rate is the saved fraction.
        out["sparsity_saving"] = float(1.0 - spike_rate)
    return out


def confusion_table(y_true, y_pred, class_names: Sequence[str]) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    w = max(10, max(len(c) for c in class_names) + 2)
    head = " " * w + "".join(f"{c[:w - 1]:>{w}}" for c in class_names)
    rows = [f"{'true \\ pred':>{w}}" + head[w:]]
    for i, c in enumerate(class_names):
        rows.append(f"{c[:w - 1]:>{w}}" + "".join(f"{v:>{w},}" for v in cm[i]))
    return "\n".join(rows)


def report(y_true, y_pred, class_names: Sequence[str]) -> str:
    return classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                 target_names=list(class_names), digits=4,
                                 zero_division=0)


def format_metrics(m: Dict[str, float], keys: Optional[List[str]] = None) -> str:
    keys = keys or ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "mcc"]
    parts = [f"{k}={m[k]:.4f}" for k in keys if k in m and isinstance(m[k], float)]
    return "  ".join(parts)
