"""Small shared helpers."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def add_repo_root() -> None:
    """Let scripts/ run without installing the package."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"[io]    saved -> {path}")


def banner(text: str, char: str = "=", width: int = 78) -> str:
    return f"\n{char * width}\n{text}\n{char * width}"


def results_table(rows) -> str:
    """rows: [{'model':..,'family':..,'metrics':{..}}, ...]"""
    cols = [("model", 22), ("family", 10), ("accuracy", 10), ("precision", 10),
            ("recall", 9), ("f1", 9), ("fpr", 9), ("roc_auc", 9)]
    head = "".join(f"{c:<{w}}" if c in ("model", "family") else f"{c:>{w}}"
                   for c, w in cols)
    out = [head, "-" * len(head)]
    for r in rows:
        m = r.get("metrics", {})
        line = f"{str(r.get('model'))[:21]:<22}{str(r.get('family'))[:9]:<10}"
        for c, w in cols[2:]:
            v = m.get(c)
            line += f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        out.append(line)
    return "\n".join(out)
