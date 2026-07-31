"""Measurement layer: coverage-aware empirical-Bayes editing estimation.

The observation model for a locus is beta-binomial. Conditional on the latent
editing fraction pi, the edited-read count is Binomial(n, pi); placing a shared
Beta(alpha, beta) prior on pi (estimated across loci -- *empirical* Bayes) gives a
conjugate Beta posterior. This is the single most Bayesian part of ADARIA: it
pulls low-coverage loci toward the genome-wide prior so a 1/1 = 100% observation
is not trusted like a 200/1000 = 20% observation.

A single-library pilot cannot identify biological-replicate dispersion, so this
module estimates one shared prior across loci and reports that limitation rather
than fabricating replicate-level precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class BetaPrior:
    """A Beta(alpha, beta) prior on an editing fraction."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        return self.alpha + self.beta


def fit_beta_prior(
    edited: np.ndarray,
    total: np.ndarray,
    min_concentration: float = 2.0,
    max_concentration: float = 10_000.0,
) -> BetaPrior:
    """Method-of-moments Beta prior with approximate sampling-noise removal.

    The observed variance of per-locus editing fractions mixes true biological
    spread with binomial sampling noise; we subtract an estimate of the sampling
    component so the prior reflects latent spread, then clip the concentration to
    a sane range. Requires >= 2 covered loci.
    """
    edited = np.asarray(edited, dtype=float)
    total = np.asarray(total, dtype=float)
    valid = np.isfinite(edited) & np.isfinite(total) & (total > 0) & (edited >= 0)
    edited = np.minimum(edited[valid], total[valid])
    total = total[valid]
    if len(total) < 2:
        raise ValueError("At least two covered loci are required to fit a beta prior")

    mean = float((edited.sum() + 0.5) / (total.sum() + 1.0))
    fraction = edited / total
    observed_variance = float(np.var(fraction, ddof=1))
    sampling_variance = float(np.mean(mean * (1.0 - mean) / total))
    latent_variance = max(observed_variance - sampling_variance, 1e-10)
    concentration = mean * (1.0 - mean) / latent_variance - 1.0
    concentration = float(np.clip(concentration, min_concentration, max_concentration))
    return BetaPrior(
        alpha=max(mean * concentration, 1e-6),
        beta=max((1.0 - mean) * concentration, 1e-6),
    )


def beta_posterior(
    edited: np.ndarray, total: np.ndarray, prior: BetaPrior
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance of each latent editing fraction.

    Conjugate update: posterior = Beta(alpha + edited, beta + total - edited).
    Returns (mean, variance) arrays, one entry per locus.
    """
    edited = np.asarray(edited, dtype=float)
    total = np.asarray(total, dtype=float)
    edited = np.minimum(np.maximum(edited, 0.0), np.maximum(total, 0.0))
    a = prior.alpha + edited
    b = prior.beta + np.maximum(total - edited, 0.0)
    mean = a / (a + b)
    variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return mean, variance


def contrast_posteriors(
    edited_a: np.ndarray,
    total_a: np.ndarray,
    prior_a: BetaPrior,
    edited_b: np.ndarray,
    total_b: np.ndarray,
    prior_b: BetaPrior,
) -> dict[str, np.ndarray]:
    """Normal-approximate posterior contrast (condition A minus condition B).

    Returns per-locus posterior means, the shrunken difference (delta), its
    standard error, the z-score (delta/se; the coverage-weighted quantity that
    downstream ranking uses), and P(A > B).
    """
    mean_a, var_a = beta_posterior(edited_a, total_a, prior_a)
    mean_b, var_b = beta_posterior(edited_b, total_b, prior_b)
    delta = mean_a - mean_b
    standard_error = np.sqrt(np.maximum(var_a + var_b, 1e-16))
    z = delta / standard_error
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "standard_error": standard_error,
        "z": z,
        "probability_a_gt_b": ndtr(z),
    }
