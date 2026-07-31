# ADARIA — ADAR Isoform Activity

**Estimate ADAR1 p150 vs p110 RNA-editing activity from bulk RNA-seq**, with
calibrated uncertainty and principled abstention when the data cannot separate
the isoforms.

ADAR1's two isoforms do different jobs: **p110** is nuclear and constitutive,
while the interferon-inducible **p150** is largely cytoplasmic and edits the
double-stranded RNA that MDA5 would otherwise sense as non-self. Existing
genome-wide summaries (AEI, CEI) return **one number per sample** — no isoform
resolution and no uncertainty. ADARIA returns both activities separately:

| output | meaning |
|---|---|
| `a150` | p150 editing activity |
| `a110` | p110 editing activity |
| `se_*` | uncertainty on each estimate |
| `abstain` | `True` when the isoforms cannot be separated |

Activity is on a calibrated scale: **`a = 1`** means "as active as in the
reference rescue experiment"; **`a = 0`** means no catalytic activity (knockout).

---

## Install

```bash
pip install git+https://github.com/shizhuom/ADARIA.git
# or, from a checkout:
git clone https://github.com/shizhuom/ADARIA.git && cd ADARIA && pip install .
```

Core dependencies are just `numpy`, `pandas`, `scipy`. The default HEK293T
signature ships **inside the package** — nothing to download.

## Quick start

```python
from adaria import ADARIA

iso = ADARIA.default()          # packaged HEK293T signature
iso.signature.report()          # ALWAYS check the signature is usable
print(iso.estimate("sample.sites.tsv"))
```

```
ActivityEstimate(sample)
  a150 = 0.3283  (SE 0.0017)
  a110 = 0.3806  (SE 0.0016)
  a150 - a110 = -0.0523  (SE 0.0032)
  anchors used: 36,224   signature: hek293t
```

Full walkthrough: **[`examples/example.ipynb`](examples/example.ipynb)** — runs
out of the box on the bundled HeLa example data.

## Input

ADARIA consumes an **editome**: per-site edited / unedited read counts with
columns `chr, pos, ref_count, edit_count`. Generate one from a BAM:

```bash
python -m adaria.pileup_sites --plusbam s.bam --minusbam s.bam \
       --sites known_AtoI_sites.bed --out s.sites.tsv --minbq 25
```

Any caller works as long as the column names match.

## How it works

1. **Signature** — from a controlled p150-only / p110-only rescue experiment,
   empirical-Bayes (beta-binomial) shrinkage gives each locus a fingerprint
   `(s150, s110)`: how strongly it responds to each isoform. Built once.
2. **Inversion** — for a new sample, solve for the single pair `(a150, a110)`
   that best explains the genome-wide editing pattern (a concave 2-parameter
   binomial likelihood → unique optimum); a Laplace approximation supplies the
   uncertainty and the abstention rule.

The discriminating signal is the **shape** of editing along the isoform-preference
axis: its *height* gives `a150 + a110` (≈ what AEI sees), its *slope* gives
`a150 − a110` (what AEI cannot see).

See [`docs/METHOD.md`](docs/METHOD.md) for the model and math.

## Validation

Three pre-registered gates, all on public data
([`docs/REPRODUCE.md`](docs/REPRODUCE.md)):

| Gate | Question | Result |
|---|---|---|
| **0** Identifiability | Is `a150 − a110` recoverable despite collinear fingerprints? | ✅ RMSE 0.031 under realistic noise vs signal ~1.0 |
| **1** Cross-system transfer | Does a HEK293T signature work in a different cell type? | ✅ AUC 0.808 vs 0.784 (raw two-anchor rule) on held-out NPC |
| **2** Genetic double-dissociation | Do the estimates respond correctly to knockouts? | ✅ p150-KO drops `a150` selectively (0.66–0.96); ADAR1-KO and catalytic-dead E912A drop both to 0 |

## Using your own signature

```python
from adaria import build_signature_from_table, ADARIA

sig = build_signature_from_table("my_rescue.tsv", min_edited_reads=30)
sig.report()                     # must be OK/WARN — never use a FAIL signature
iso = ADARIA(sig)
```

⚠️ **Always run `report()`.** An information-poor signature yields *confidently
wrong* activities (wild-type read as zero, knockout as non-zero) — uncertainty
alone does not catch this, because it is bias rather than variance. The
`min_edited_reads` filter matters most: isoform information scales with
`n·π(1−π)`, so a deeply covered but barely edited locus adds only noise, and
filtering on **coverage** does not remove those.

⚠️ **Activities from different signatures are not comparable** (`a = 1` is
defined by *that* signature's reference experiment). The signature used is
recorded in `signature_id`.

## Known limitations

- **Panel-defined.** Activity is measured on the signature's loci — as AEI is
  defined on Alu and CEI on 3′UTR inverted Alu. Random loss of loci costs
  precision, not accuracy; *systematic* loss (e.g. by expression) can shift
  estimates, so compare samples on a common set of covered anchors.
- **Reported SEs are optimistic** (~8×): they come from a Laplace approximation
  with a binomial likelihood and a signature treated as exact. Use them for
  relative comparison and calibrate abstention thresholds accordingly.
- **The signature is a plug-in** — its own estimation error is not propagated.
- **`b0`** (zero-activity editing baseline, default 0.005) is a fixed constant;
  lower it for signatures with much lower editing (`report()` will warn).

## Repository layout

```
src/adaria/          the package (+ packaged default signature)
examples/            example.ipynb + small bundled HeLa editomes
tests/               unit tests  (pytest)
pipeline/            scripts that reproduce the paper end to end
docs/METHOD.md       the statistical model
docs/REPRODUCE.md    how to regenerate every result
```

## Citation

See [`CITATION.cff`](CITATION.cff). Paper in preparation.

## License

MIT — see [`LICENSE`](LICENSE).
