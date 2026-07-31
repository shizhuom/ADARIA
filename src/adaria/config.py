"""Load the single-source-of-truth config and resolve project-relative paths."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read config/config.yaml. Relative `paths.*` are resolved against the repo root."""
    import yaml  # lazy: keeps the core library importable without pyyaml

    path = Path(path) if path else _REPO_ROOT / "config" / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    root = Path(cfg.get("paths", {}).get("root", _REPO_ROOT))
    resolved = {}
    for key, val in cfg.get("paths", {}).items():
        p = Path(val)
        resolved[key] = str(p if p.is_absolute() else root / p)
    cfg["resolved_paths"] = resolved
    return cfg
