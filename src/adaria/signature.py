"""Signature layer: build the per-locus p150 / p110 fingerprint.

Input is an author-provided genome-wide index table (one row per editing cluster)
carrying per-condition read counts:

    P150_cov / P150_Gs  ->  tot_150 / edit_150   (coverage / edited reads, p150 arm)
    P110_cov / P110_Gs  ->  tot_110 / edit_110

For each cluster j we compute empirical-Bayes shrunken editing under each isoform
(theta150_j, theta110_j) and the logit-scale "susceptibility" relative to a fixed
catalytic-null baseline b0:

    s150_j = logit(theta150_j) - logit(b0)      s110_j = logit(theta110_j) - logit(b0)

The pair (s150_j, s110_j) is the fingerprint used to invert new samples. The
discriminating signal s150_j - s110_j = logit(theta150_j) - logit(theta110_j) is
exactly the p150/p110 editing preference and is independent of b0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .editing import BetaPrior, beta_posterior, fit_beta_prior
from .utils import expit, logit

_NEEDED = {"seqnames", "start", "end", "N_sites",
           "P150_cov", "P150_Gs", "P110_cov", "P110_Gs"}


def load_index_table(path: str) -> pd.DataFrame:
    """Load an *_vs_* index table, handling both on-disk layouts.

    HEK150_vs_HEK110 starts at `seqnames`; NPC150_vs_NPC110 was written by R with
    row.names=TRUE, leaving an unnamed leading cluster-key column. We detect the
    shift (a `seqnames` value that looks like `chr1_1032168_1033084`) and re-read
    with the key as the index. Returns standardized columns:
    chr, start, end, n_sites, edit_150, tot_150, edit_110, tot_110 (+ indices/pvals).
    """
    df = pd.read_csv(path, sep="\t")
    if "seqnames" not in df.columns or df["seqnames"].astype(str).str.contains("_").any():
        df = pd.read_csv(path, sep="\t", index_col=0)
    missing = _NEEDED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")

    out = pd.DataFrame({
        "chr": df["seqnames"].astype(str),
        "start": df["start"].astype(int),
        "end": df["end"].astype(int),
        "n_sites": df["N_sites"].astype(int),
        "edit_150": df["P150_Gs"].astype(float),
        "tot_150": df["P150_cov"].astype(float),
        "edit_110": df["P110_Gs"].astype(float),
        "tot_110": df["P110_cov"].astype(float),
    })
    for opt in ("P150_index", "P110_index", "pval", "qval"):
        if opt in df.columns:
            out[opt.lower()] = pd.to_numeric(df[opt], errors="coerce")
    return out.reset_index(drop=True)


@dataclass
class Signature:
    """A fitted p150/p110 fingerprint plus the empirical-Bayes priors it used."""

    table: pd.DataFrame        # per-anchor: chr,start,end,...,s150,s110,eta,min_cov
    prior150: BetaPrior
    prior110: BetaPrior
    b0: float

    def report(self, verbose: bool = True) -> dict:
        """Usability diagnostics for this signature. ALWAYS check before use.

        A signature can be built from any rescue-style table, but not every table
        yields a usable one. Empirically, an information-poor signature (loci well
        covered but barely edited) produces confidently WRONG activities -- e.g.
        wild-type read as zero activity and knockout as non-zero. Uncertainty alone
        does not catch this, so these structural checks are required.

        Decisive checks (thresholds from the NPC failure/repair experiment):
          theta_over_b0     median editing vs the zero-activity baseline; <1 means
                            the baseline sits above typical editing -> logic inverts
          frac_negative_s   fraction of loci with s150<0 or s110<0
          median_edited     median edited reads per locus (information content)
        """
        t = self.table
        med_theta = float(np.median(np.concatenate([t.theta150, t.theta110])))
        frac_neg = float(((t.s150 < 0) | (t.s110 < 0)).mean())
        med_edited = float(np.median(t.edit_150 + t.edit_110))
        rep = identifiability_report(self)
        rep.update({
            "b0": self.b0,
            "median_theta": med_theta,
            "theta_over_b0": med_theta / self.b0 if self.b0 > 0 else float("inf"),
            "frac_negative_s": frac_neg,
            "median_edited_reads": med_edited,
            "prior150_mean": self.prior150.mean,
            "prior110_mean": self.prior110.mean,
        })

        problems, warnings = [], []
        if rep["theta_over_b0"] < 1.0:
            problems.append(f"median editing ({med_theta:.4f}) is BELOW b0 ({self.b0}) "
                            f"-- lower b0 or filter on edited reads")
        if frac_neg > 0.40:
            problems.append(f"{frac_neg:.0%} of loci have negative susceptibility")
        elif frac_neg > 0.20:
            warnings.append(f"{frac_neg:.0%} of loci have negative susceptibility")
        if med_edited < 10:
            warnings.append(f"median edited reads is only {med_edited:.0f} "
                            f"-- consider min_edited_reads>=30")
        if rep["n_anchors"] < 1000:
            warnings.append(f"only {rep['n_anchors']} anchors")
        rep["verdict"] = "FAIL" if problems else ("WARN" if warnings else "OK")
        rep["problems"] = problems
        rep["warnings"] = warnings

        if verbose:
            print(f"Signature diagnostics  [{rep['verdict']}]")
            print(f"  anchors                {rep['n_anchors']:,}")
            print(f"  median editing / b0    {rep['theta_over_b0']:.1f}x   "
                  f"(median theta={med_theta:.4f}, b0={self.b0})")
            print(f"  loci with s<0          {frac_neg:.1%}")
            print(f"  median edited reads    {med_edited:.0f}")
            print(f"  collinearity(s150,s110){rep['cov_weighted_corr_s150_s110']:.3f}"
                  f"   Fisher cond. {rep['fisher_condition_number']:.1f}")
            for p in problems:
                print(f"  FAIL: {p}")
            for w in warnings:
                print(f"  WARN: {w}")
        return rep


def build_signature(
    table: pd.DataFrame,
    min_condition_coverage: int = 10,
    min_edited_reads: int = 0,
    b0: float = 0.005,
    eb_min_concentration: float = 2.0,
    eb_max_concentration: float = 10_000.0,
) -> Signature:
    """Fit the empirical-Bayes fingerprint from a standardized index table.

    `min_edited_reads` filters on the number of EDITED reads (summed over both
    isoform arms), not on coverage. This matters: information about which isoform
    edited a locus scales with n*pi*(1-pi), so a deeply covered but essentially
    unedited cluster (pi ~ 0.002) carries almost no isoform information and only
    injects noise into s150/s110. Filtering on coverage alone does not remove
    these; filtering on edited reads does. See docs/PLAN.md.
    """
    t = table[(table.tot_150 >= min_condition_coverage)
              & (table.tot_110 >= min_condition_coverage)]
    if min_edited_reads > 0:
        t = t[(t.edit_150 + t.edit_110) >= min_edited_reads]
    t = t.copy().reset_index(drop=True)
    if len(t) < 2:
        raise ValueError("need >= 2 clusters passing the coverage/editing filters")

    prior150 = fit_beta_prior(t.edit_150.to_numpy(), t.tot_150.to_numpy(),
                              eb_min_concentration, eb_max_concentration)
    prior110 = fit_beta_prior(t.edit_110.to_numpy(), t.tot_110.to_numpy(),
                              eb_min_concentration, eb_max_concentration)
    theta150, _ = beta_posterior(t.edit_150.to_numpy(), t.tot_150.to_numpy(), prior150)
    theta110, _ = beta_posterior(t.edit_110.to_numpy(), t.tot_110.to_numpy(), prior110)

    eta = float(logit(b0))
    t["theta150"] = theta150
    t["theta110"] = theta110
    t["s150"] = logit(theta150) - eta
    t["s110"] = logit(theta110) - eta
    t["eta"] = eta
    t["min_cov"] = np.minimum(t.tot_150, t.tot_110)
    return Signature(table=t, prior150=prior150, prior110=prior110, b0=b0)


def identifiability_report(sig: Signature, operating_point=(1.0, 1.0)) -> dict:
    """Collinearity + Fisher-conditioning diagnostics for the signature.

    A high `cov_weighted_corr` and large `fisher_condition_number` mean the two
    columns are nearly parallel; `weakest_axis_align_contrast` near 1 confirms the
    poorly-identified direction is exactly a150-a110 (the quantity we want).
    """
    t = sig.table
    s150 = t.s150.to_numpy(); s110 = t.s110.to_numpy(); eta = t.eta.to_numpy()
    w = t.min_cov.to_numpy()

    m150 = np.average(s150, weights=w); m110 = np.average(s110, weights=w)
    c150 = s150 - m150; c110 = s110 - m110
    corr = float(np.average(c150 * c110, weights=w)
                 / np.sqrt(np.average(c150 ** 2, weights=w) * np.average(c110 ** 2, weights=w)))

    a150, a110 = operating_point
    pi = np.clip(expit(eta + s150 * a150 + s110 * a110), 1e-4, 1 - 1e-4)
    fw = w * pi * (1 - pi)
    S = np.column_stack([s150, s110])
    info = S.T @ (fw[:, None] * S)
    evals, evecs = np.linalg.eigh(info)
    contrast_dir = np.array([1.0, -1.0]) / np.sqrt(2.0)
    return {
        "n_anchors": int(len(t)),
        "cov_weighted_corr_s150_s110": corr,
        "fisher_condition_number": float(evals[1] / max(evals[0], 1e-12)),
        "weakest_axis_align_contrast": float(abs(evecs[:, 0] @ contrast_dir)),
    }


def save_signature(sig: Signature, table_path: str, meta_path: str) -> None:
    """Persist a signature: anchor table (TSV) + priors/b0 (JSON)."""
    import json

    sig.table.to_csv(table_path, sep="\t", index=False)
    with open(meta_path, "w") as fh:
        json.dump({
            "b0": sig.b0,
            "prior150": [sig.prior150.alpha, sig.prior150.beta],
            "prior110": [sig.prior110.alpha, sig.prior110.beta],
        }, fh, indent=2)


def load_signature(table_path: str, meta_path: str) -> Signature:
    """Reload a signature saved by `save_signature`."""
    import json

    table = pd.read_csv(table_path, sep="\t")
    with open(meta_path) as fh:
        meta = json.load(fh)
    return Signature(table=table, b0=meta["b0"],
                     prior150=BetaPrior(*meta["prior150"]),
                     prior110=BetaPrior(*meta["prior110"]))
