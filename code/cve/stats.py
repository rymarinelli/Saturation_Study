"""
stats.py — uncertainty quantification for the CVE section.

Shares   : Beta-Binomial posterior with a Jeffreys Beta(1/2, 1/2) prior; posterior
           mean + equal-tailed credible interval; differences/ratios between
           periods by Monte Carlo from the two independent posteriors.
CVSS     : CVSS base scores live on a 0.1 grid (101 values), so each group is a
           count histogram. The distribution of all pairwise differences
           x_i - y_j is the cross-correlation of the two histograms, which gives
           the Hodges-Lehmann shift (median pairwise difference) and the
           probability of superiority P(X>Y) + 1/2 P(X=Y) exactly, in O(101^2).
           Percentile bootstrap: multinomial resampling of each histogram, which
           is equivalent to resampling CVEs with replacement within each group.
           Mann-Whitney U p-value from scipy (asymptotic, tie-corrected).
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import stats as st

import config as C

GRID = 101  # 0.0, 0.1, ..., 10.0


def rng_for(label: str) -> np.random.Generator:
    """Independent, order-free RNG stream per named quantity."""
    return np.random.default_rng([C.SEED, zlib.crc32(label.encode())])


# ─────────────────────────────────────────────────────────────────────────────
# Shares
# ─────────────────────────────────────────────────────────────────────────────
def beta_posterior(k: int, n: int, prior=C.JEFFREYS, cred: float = C.CRED) -> dict:
    a, b = k + prior[0], n - k + prior[1]
    lo, hi = st.beta.ppf([(1 - cred) / 2, 1 - (1 - cred) / 2], a, b)
    return {"k": int(k), "n": int(n), "mean": a / (a + b), "lo": float(lo), "hi": float(hi)}


def share_contrast(k1: int, n1: int, k2: int, n2: int, label: str,
                   prior=C.JEFFREYS, cred: float = C.CRED, n_mc: int = C.N_MC) -> dict:
    """Posterior of p2 - p1 and p2 / p1 (period 2 vs period 1)."""
    rng = rng_for(label)
    p1 = rng.beta(k1 + prior[0], n1 - k1 + prior[1], n_mc)
    p2 = rng.beta(k2 + prior[0], n2 - k2 + prior[1], n_mc)
    q = [(1 - cred) / 2, 1 - (1 - cred) / 2]
    d, r = p2 - p1, p2 / p1
    return {"diff_mean": float(d.mean()), "diff_lo": float(np.quantile(d, q[0])),
            "diff_hi": float(np.quantile(d, q[1])),
            "ratio_median": float(np.median(r)), "ratio_lo": float(np.quantile(r, q[0])),
            "ratio_hi": float(np.quantile(r, q[1])),
            "p_increase": float(np.mean(d > 0))}


# ─────────────────────────────────────────────────────────────────────────────
# CVSS location shift
# ─────────────────────────────────────────────────────────────────────────────
def to_hist(scores) -> np.ndarray:
    idx = np.rint(np.asarray(scores, float) * 10).astype(int)
    if idx.size and (idx.min() < 0 or idx.max() > GRID - 1):
        raise ValueError("CVSS score outside [0, 10]")
    return np.bincount(idx, minlength=GRID).astype(np.int64)


def _pair_weights(ha: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """w[k] = #pairs with x - y = (k - 100) / 10."""
    return np.convolve(ha, hb[::-1])


def _weighted_median(w: np.ndarray) -> float:
    W = int(w.sum())
    cum = np.cumsum(w)
    r1, r2 = (W - 1) // 2, W // 2
    k1 = int(np.searchsorted(cum, r1, side="right"))
    k2 = int(np.searchsorted(cum, r2, side="right"))
    return ((k1 - (GRID - 1)) + (k2 - (GRID - 1))) / 20.0


def hl_and_psup(ha: np.ndarray, hb: np.ndarray) -> tuple[float, float, float]:
    """(Hodges-Lehmann shift, P(X>Y)+0.5P(X=Y), P(X=Y)) for histograms a (X) and b (Y)."""
    w = _pair_weights(ha, hb)
    W = w.sum()
    mid = GRID - 1
    p_gt = w[mid + 1:].sum() / W
    p_eq = w[mid] / W
    return _weighted_median(w), float(p_gt + 0.5 * p_eq), float(p_eq)


def cvss_shift(x, y, label: str, n_boot: int = C.N_BOOT, cred: float = C.CRED) -> dict:
    """Location shift of X (LLM-related) relative to Y (all other CVEs)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ha, hb = to_hist(x), to_hist(y)
    hl, psup, peq = hl_and_psup(ha, hb)

    rng = rng_for(label)
    ba = rng.multinomial(len(x), ha / ha.sum(), size=n_boot)
    bb = rng.multinomial(len(y), hb / hb.sum(), size=n_boot)
    boot = np.array([hl_and_psup(ba[i], bb[i])[:2] for i in range(n_boot)])
    q = [(1 - cred) / 2, 1 - (1 - cred) / 2]
    hl_lo, hl_hi = np.quantile(boot[:, 0], q)
    ps_lo, ps_hi = np.quantile(boot[:, 1], q)

    mw = st.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return {
        "n_llm": len(x), "n_other": len(y),
        "median_llm": float(np.median(x)), "median_other": float(np.median(y)),
        "hl_shift": hl, "hl_lo": float(hl_lo), "hl_hi": float(hl_hi),
        "hl_boot_distinct": int(len(np.unique(boot[:, 0]))),
        "p_superiority": psup, "psup_lo": float(ps_lo), "psup_hi": float(ps_hi),
        "rank_biserial": 2 * psup - 1,
        "p_tie_pairs": peq,
        "mw_U": float(mw.statistic), "mw_p": float(mw.pvalue),
    }


def hl_bruteforce(x, y) -> float:
    """Reference implementation for verify.py."""
    d = (np.asarray(x, float)[:, None] - np.asarray(y, float)[None, :]).ravel()
    return float(np.median(np.round(d, 1)))
