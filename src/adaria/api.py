"""User-facing API.

Typical use -- estimate p150/p110 activity for a new sample:

    from adaria import ADARIA
    iso = ADARIA.default()              # the paper's HEK293T signature
    iso.signature.report()                # always inspect the signature first
    res = iso.estimate("sample.sites.tsv")
    print(res)                            # a150, a110, SE, abstain

Bring your own signature (any p150-vs-p110 rescue table with read counts):

    from adaria import build_signature_from_table, ADARIA
    sig = build_signature_from_table("my_rescue.tsv", min_edited_reads=30)
    sig.report()                          # MUST be OK/WARN, not FAIL
    iso = ADARIA(sig)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .inversion import aggregate_editome_to_anchors, invert_sample, should_abstain
from .signature import Signature, build_signature, load_index_table, load_signature

# The default HEK293T signature ships inside the package, so `pip install adaria`
# is enough -- no repo checkout or data download needed.
_PKG_DATA = Path(__file__).resolve().parent / "data"
_DEFAULT_SIG = _PKG_DATA / "signature_hek.tsv.gz"
_DEFAULT_META = _PKG_DATA / "signature_hek.meta.json"


@dataclass
class ActivityEstimate:
    """Estimated isoform activities for one sample."""

    a150: float
    a110: float
    se_a150: float
    se_a110: float
    se_contrast: float
    n_anchors: int
    abstain: bool
    signature_id: str
    sample: str = ""

    def __repr__(self) -> str:
        flag = "  [ABSTAIN: contrast not identifiable]" if self.abstain else ""
        return (f"ActivityEstimate({self.sample or 'sample'})\n"
                f"  a150 = {self.a150:.4f}  (SE {self.se_a150:.4f})\n"
                f"  a110 = {self.a110:.4f}  (SE {self.se_a110:.4f})\n"
                f"  a150 - a110 = {self.a150 - self.a110:+.4f}  (SE {self.se_contrast:.4f})\n"
                f"  anchors used: {self.n_anchors:,}   signature: {self.signature_id}{flag}")

    def to_dict(self) -> dict:
        return asdict(self)


def build_signature_from_table(path, min_condition_coverage: int = 10,
                               min_edited_reads: int = 30, b0: float = 0.005) -> Signature:
    """Build a signature from a p150-vs-p110 index table (with read counts).

    The table needs per-cluster coordinates plus P150_cov/P150_Gs and
    P110_cov/P110_Gs (coverage and edited reads for each isoform arm).
    """
    return build_signature(load_index_table(str(path)),
                           min_condition_coverage=min_condition_coverage,
                           min_edited_reads=min_edited_reads, b0=b0)


class ADARIA:
    """Estimate per-sample ADAR1 p150/p110 editing activity from an editome."""

    def __init__(self, signature: Signature, signature_id: str = "custom",
                 abstain_se_contrast: float | None = None):
        self.signature = signature
        self.signature_id = signature_id
        self.abstain_se_contrast = abstain_se_contrast

    @classmethod
    def default(cls, abstain_se_contrast: float | None = None) -> "ADARIA":
        """Load the packaged HEK293T signature used in the paper."""
        if not _DEFAULT_SIG.exists():
            raise FileNotFoundError(
                f"packaged signature missing at {_DEFAULT_SIG} -- reinstall adaria, "
                "or pass your own Signature to ADARIA(...)")
        sig = load_signature(str(_DEFAULT_SIG), str(_DEFAULT_META))
        return cls(sig, signature_id="hek293t", abstain_se_contrast=abstain_se_contrast)

    # ---------------------------------------------------------------- estimation
    def estimate(self, editome, sample: str = "") -> ActivityEstimate:
        """Estimate (a150, a110) from a per-site editome.

        `editome` is a path or DataFrame with columns chr, pos, ref_count,
        edit_count (the pileup output of pipeline stage 04). Sites are summed into
        the signature's anchors, then the activities are inverted.
        """
        if not isinstance(editome, pd.DataFrame):
            sample = sample or Path(editome).stem
            editome = pd.read_csv(editome, sep="\t")
        edit, tot = aggregate_editome_to_anchors(editome, self.signature)
        return self._finish(edit, tot, sample)

    def estimate_from_index_table(self, path, arm: str = "p150") -> ActivityEstimate:
        """Estimate from one arm of an index table (counts already per-cluster)."""
        t = load_index_table(str(path))
        key = "150" if arm == "p150" else "110"
        s = t[["chr", "start", "end", f"edit_{key}", f"tot_{key}"]].rename(
            columns={f"edit_{key}": "edit", f"tot_{key}": "tot"})
        m = self.signature.table.merge(s, on=["chr", "start", "end"], how="left")
        return self._finish(m["edit"].fillna(0).to_numpy(), m["tot"].fillna(0).to_numpy(),
                            f"{Path(path).stem}:{arm}")

    def estimate_many(self, editomes: dict) -> pd.DataFrame:
        """Estimate for several samples -> tidy DataFrame. {name: path_or_df}."""
        return pd.DataFrame([self.estimate(v, sample=k).to_dict()
                             for k, v in editomes.items()])

    def _finish(self, edit, tot, sample: str) -> ActivityEstimate:
        r = invert_sample(edit, tot, self.signature)
        return ActivityEstimate(
            a150=r.a150, a110=r.a110, se_a150=r.se_a150, se_a110=r.se_a110,
            se_contrast=r.se_contrast, n_anchors=r.n_anchors,
            abstain=should_abstain(r, self.abstain_se_contrast),
            signature_id=self.signature_id, sample=sample)
