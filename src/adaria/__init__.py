"""ADARIA: coverage-aware estimation of ADAR1 p150/p110 editing activity.

Quick start
-----------
    from adaria import ADARIA
    iso = ADARIA.default()          # paper's HEK293T signature
    iso.signature.report()            # check the signature is usable
    print(iso.estimate("sample.sites.tsv"))

Layers
------
  editing    empirical-Bayes beta-binomial measurement (shrinkage)
  signature  per-locus p150/p110 fingerprint (s150, s110) + diagnostics
  inversion  per-sample (a150, a110) with uncertainty and abstention
  api        user-facing ADARIA class
"""
from __future__ import annotations

from .api import ActivityEstimate, ADARIA, build_signature_from_table
from .config import load_config
from .editing import BetaPrior, beta_posterior, contrast_posteriors, fit_beta_prior
from .inversion import (
    ActivityResult,
    aggregate_editome_to_anchors,
    invert,
    invert_sample,
    should_abstain,
)
from .signature import (
    Signature,
    build_signature,
    identifiability_report,
    load_index_table,
    load_signature,
    save_signature,
)
from .utils import expit, logit

__version__ = "0.1.0"

__all__ = [
    # user-facing
    "ADARIA", "ActivityEstimate", "build_signature_from_table",
    # signature layer
    "Signature", "build_signature", "load_index_table", "load_signature",
    "save_signature", "identifiability_report",
    # inversion layer
    "ActivityResult", "invert", "invert_sample", "should_abstain",
    "aggregate_editome_to_anchors",
    # measurement layer
    "BetaPrior", "fit_beta_prior", "beta_posterior", "contrast_posteriors",
    # misc
    "load_config", "logit", "expit", "__version__",
]
