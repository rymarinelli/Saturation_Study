"""
verify.py — correctness checks for the CVE pipeline (not needed to reproduce the paper).

    python code/cve/verify.py stats
        Histogram Hodges-Lehmann / P(superiority) vs brute force; Jeffreys CrI vs
        scipy.stats.beta.

    python code/cve/verify.py legacy --nvd OLD_CLONE [--old-csv llm_cves.csv]
        Port fidelity: with the notebook's original settings (ID directories
        CVE-2022..CVE-2026 only, rejected records kept, keyword filter only) the
        scanner must reproduce the notebook's counts (1,767 kept / 169,950 not
        kept on the 30 Jun 2026 clone) and, if given, the exact kept ID set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan  # noqa: E402
import stats as S  # noqa: E402


def check_stats():
    rng = np.random.default_rng(0)
    grid = np.round(np.arange(0, 10.01, 0.1), 1)
    for trial in range(5):
        pa = rng.dirichlet(np.ones(len(grid)) * 0.3)
        pb = rng.dirichlet(np.ones(len(grid)) * 0.3)
        x = rng.choice(grid, size=rng.integers(50, 2000), p=pa)
        y = rng.choice(grid, size=rng.integers(50, 2000), p=pb)
        hl, psup, _ = S.hl_and_psup(S.to_hist(x), S.to_hist(y))
        hl_ref = S.hl_bruteforce(x, y)
        d = x[:, None] - y[None, :]
        psup_ref = float((d > 1e-9).mean() + 0.5 * (np.abs(d) < 1e-9).mean())
        u = st.mannwhitneyu(x, y).statistic / (len(x) * len(y))
        assert abs(hl - hl_ref) < 1e-9, (trial, hl, hl_ref)
        assert abs(psup - psup_ref) < 1e-9 and abs(psup - u) < 1e-9, (trial, psup, psup_ref, u)
    for k, n in ((0, 100), (7, 1000), (1767, 171717)):
        b = S.beta_posterior(k, n)
        lo, hi = st.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
        assert np.isclose(b["lo"], lo) and np.isclose(b["hi"], hi)
    print("stats: OK (HL, P(superiority) == brute force / Mann-Whitney U; Jeffreys CrI == scipy)")


def check_legacy(nvd: Path, old_csv: Path | None):
    df = scan.load_or_scan(nvd, tag="legacy", dirs=[f"CVE-{y}" for y in range(2022, 2027)])
    d = df[df["has_en_desc"]]
    kept = d["keyword"].ne("")
    print(f"legacy: kept={int(kept.sum()):,}  not kept={int((~kept).sum()):,}  (notebook: 1,767 / 169,950)")
    if old_csv is not None:
        old = set(pd.read_csv(old_csv, usecols=["cve_id"])["cve_id"])
        new = set(d.loc[kept, "cve_id"])
        print(f"legacy: ID sets identical: {old == new}  (only old: {len(old - new)}, only new: {len(new - old)})")
        assert old == new
    assert int(kept.sum()) == 1767 and int((~kept).sum()) == 169950
    print("legacy: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["stats", "legacy"])
    ap.add_argument("--nvd", type=Path)
    ap.add_argument("--old-csv", type=Path)
    a = ap.parse_args()
    check_stats() if a.what == "stats" else check_legacy(a.nvd, a.old_csv)
