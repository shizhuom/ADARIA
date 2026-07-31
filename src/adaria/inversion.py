"""Inversion layer: the ADARIA estimator.

Given a sample's per-anchor edited/total counts and a fitted signature, recover the
latent isoform activities (a150, a110) by maximizing the binomial likelihood

    pi_j(a) = expit( eta_j + s150_j * a150 + s110_j * a110 )
    -logL   = -sum_j [ e_j log pi_j + (n_j - e_j) log(1 - pi_j) ]

The point estimate is maximum likelihood (a convex problem in the linear
predictor). A Laplace approximation at the optimum (the Fisher/observed
information = the Hessian) yields the uncertainty used for the abstention rule --
in particular the SE of the contrast (a150 - a110), which is the ill-conditioned
direction when the two susceptibility columns are collinear.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .utils import CLIP, expit


@dataclass
class ActivityResult:
    a150: float
    a110: float
    se_a150: float
    se_a110: float
    se_contrast: float        # SE of (a150 - a110): the abstention-relevant quantity
    n_anchors: int
    converged: bool


def invert(
    edit: np.ndarray,
    tot: np.ndarray,
    s150: np.ndarray,
    s110: np.ndarray,
    eta: np.ndarray | float,
    bounds: tuple[float, float] = (0.0, 5.0),
) -> ActivityResult:
    """Maximum-likelihood (a150, a110) with a Laplace-approximate posterior SE."""
    edit = np.asarray(edit, float)
    tot = np.asarray(tot, float)
    s150 = np.asarray(s150, float)
    s110 = np.asarray(s110, float)
    eta = np.asarray(eta, float)
    if eta.ndim == 0:
        eta = np.full(len(edit), float(eta))

    keep = tot >= 1
    e, n = edit[keep], tot[keep]
    a1, a2, et = s150[keep], s110[keep], eta[keep]
    if len(e) < 2:
        return ActivityResult(np.nan, np.nan, np.nan, np.nan, np.nan, int(keep.sum()), False)

    def nll_and_grad(a):
        pi = np.clip(expit(et + a1 * a[0] + a2 * a[1]), CLIP, 1 - CLIP)
        nll = -np.sum(e * np.log(pi) + (n - e) * np.log(1 - pi))
        resid = e - n * pi
        grad = np.array([-np.sum(resid * a1), -np.sum(resid * a2)])
        return nll, grad

    res = minimize(nll_and_grad, x0=np.array([1.0, 1.0]), jac=True,
                   method="L-BFGS-B", bounds=[bounds, bounds])
    a = res.x
    pi = np.clip(expit(et + a1 * a[0] + a2 * a[1]), CLIP, 1 - CLIP)
    w = n * pi * (1 - pi)
    hess = np.array([[np.sum(w * a1 * a1), np.sum(w * a1 * a2)],
                     [np.sum(w * a1 * a2), np.sum(w * a2 * a2)]])
    try:
        cov = np.linalg.inv(hess + 1e-9 * np.eye(2))
        se1 = float(np.sqrt(max(cov[0, 0], 0.0)))
        se2 = float(np.sqrt(max(cov[1, 1], 0.0)))
        c = np.array([1.0, -1.0])
        se_c = float(np.sqrt(max(c @ cov @ c, 0.0)))
    except np.linalg.LinAlgError:
        se1 = se2 = se_c = float("nan")
    return ActivityResult(float(a[0]), float(a[1]), se1, se2, se_c,
                          int(keep.sum()), bool(res.success))


def invert_sample(edit, tot, signature, bounds=(0.0, 5.0)) -> ActivityResult:
    """Invert a sample whose (edit, tot) arrays are aligned to signature.table rows."""
    t = signature.table
    return invert(edit, tot, t.s150.to_numpy(), t.s110.to_numpy(),
                  t.eta.to_numpy(), bounds=bounds)


def should_abstain(result: ActivityResult, se_contrast_threshold: float | None) -> bool:
    """Abstain when the contrast is not identifiable at the prespecified threshold."""
    if se_contrast_threshold is None:
        return False
    return (not np.isfinite(result.se_contrast)) or result.se_contrast > se_contrast_threshold


def aggregate_editome_to_anchors(editome, signature):
    """Sum a per-site editome into signature anchors -> (edit, tot) aligned to rows.

    `editome` is a DataFrame with columns chr, pos, edit_count, ref_count (the
    stage-04 pileup format). Used for real samples (Gate 2); the author index
    tables already provide per-cluster counts and skip this step.
    """
    t = signature.table.reset_index(drop=True)
    edit = np.zeros(len(t)); tot = np.zeros(len(t))
    by_chr = {}
    for c, g in t.groupby("chr"):
        # searchsorted requires starts sorted within each chromosome; the index
        # table is NOT coordinate-sorted, so sort here (and carry the row index).
        order = np.argsort(g.start.to_numpy(), kind="stable")
        by_chr[c] = (g.start.to_numpy()[order], g.end.to_numpy()[order], g.index.to_numpy()[order])
    for c, sub in editome.groupby("chr"):
        ce = by_chr.get(c)
        if ce is None:
            continue
        starts, ends, idx = ce
        pos = sub.pos.to_numpy()
        i = np.searchsorted(starts, pos, side="right") - 1
        ii = np.clip(i, 0, len(ends) - 1)
        ok = (i >= 0) & (ends[ii] >= pos)
        anchors = idx[ii[ok]]
        ec = sub.edit_count.to_numpy()[ok]
        rc = sub.ref_count.to_numpy()[ok]
        np.add.at(edit, anchors, ec)
        np.add.at(tot, anchors, ec + rc)
    return edit, tot
