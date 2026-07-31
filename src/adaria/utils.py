"""Numerical helpers shared across ADARIA.

Kept tiny and dependency-light so every other module can import them without
pulling in heavy machinery.
"""
from __future__ import annotations

import numpy as np

# Editing fractions are clipped away from {0,1} before the logit so that a
# perfectly un-edited or fully-edited locus does not produce +/-inf.
CLIP = 1e-6


def logit(p: np.ndarray | float, clip: float = CLIP) -> np.ndarray:
    """Log-odds transform, clipped to (clip, 1-clip)."""
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def expit(x: np.ndarray | float) -> np.ndarray:
    """Inverse logit (logistic). Input is clipped to avoid overflow."""
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))
