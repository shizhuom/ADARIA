#!/usr/bin/env python3
"""Strand-AWARE A/G quantification at known A-to-I sites.

The libraries are strand-specific (fr-firststrand / RF). Counting reads
strand-unaware dilutes editing at cis-NATs (both strands transcribed) and adds
noise elsewhere. So we split reads by transcript strand and count only the reads
from the site's own transcript strand:

  '+' site (A->G): use the +-transcript BAM, ref = #A, edit = #G
  '-' site (A->G on transcript = T->C genomic): use the --transcript BAM, ref = #T, edit = #C

Inputs are two pre-split, strand-specific BAMs (see 04_pileup_editing.sbatch).
Output TSV: chr  pos(1-based)  strand  ref_count  edit_count  coverage
"""
import argparse, sys
import pysam

def load_sites(path):
    sites = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, start, strand = f[0], int(f[1]), (f[5] if len(f) > 5 else "+")
            sites.setdefault(chrom, []).append((start, strand))
    for c in sites:
        sites[c].sort()
    return sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plusbam", required=True, help="reads from + transcript")
    ap.add_argument("--minusbam", required=True, help="reads from - transcript")
    ap.add_argument("--sites", required=True, help="BED6 chr start0 end name score strand")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minbq", type=int, default=25)
    ap.add_argument("--window", type=int, default=10_000_000)
    args = ap.parse_args()

    sites = load_sites(args.sites)
    bp = pysam.AlignmentFile(args.plusbam, "rb")
    bm = pysam.AlignmentFile(args.minusbam, "rb")
    refs = set(bp.references)
    n_out = 0
    with open(args.out, "w") as out:
        out.write("chr\tpos\tstrand\tref_count\tedit_count\tcoverage\n")
        for chrom, lst in sites.items():
            if chrom not in refs:
                continue
            clen = bp.get_reference_length(chrom)
            i = 0
            for wstart in range(0, clen, args.window):
                wend = min(wstart + args.window, clen)
                win = []
                while i < len(lst) and lst[i][0] < wend:
                    if lst[i][0] >= wstart:
                        win.append(lst[i])
                    i += 1
                if not win:
                    continue
                has_plus = any(s == "+" for _, s in win)
                has_minus = any(s == "-" for _, s in win)
                # count_coverage returns (A,C,G,T), base-quality filtered, with
                # default 'all' read filter (skips unmapped/secondary/dup/qcfail).
                covp = bp.count_coverage(chrom, wstart, wend, quality_threshold=args.minbq) if has_plus else None
                covm = bm.count_coverage(chrom, wstart, wend, quality_threshold=args.minbq) if has_minus else None
                for pos0, strand in win:
                    o = pos0 - wstart
                    if strand == "+":
                        ref_c, edit_c = covp[0][o], covp[2][o]      # A, G
                    else:
                        ref_c, edit_c = covm[3][o], covm[1][o]      # T, C
                    cov = ref_c + edit_c
                    if cov == 0:
                        continue
                    out.write(f"{chrom}\t{pos0+1}\t{strand}\t{ref_c}\t{edit_c}\t{cov}\n")
                    n_out += 1
    bp.close(); bm.close()
    sys.stderr.write(f"[pileup] wrote {n_out} covered sites to {args.out}\n")

if __name__ == "__main__":
    main()
