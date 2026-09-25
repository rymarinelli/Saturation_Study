"""
bm25.py — BM25 relevance scoring of CVE texts against the LLM lexicon.

    score(D, Q) = sum_q IDF(q) * tf(q,D)*(k1+1) / (tf(q,D) + k1*(1 - b + b*|D|/avgdl))

D = English description + CPE strings, Q = lexicon.LLM_QUERY_TERMS. Because Q is
the keyword lexicon in token form, BM25 is a graded, rarity-weighted and
length-normalized re-scoring of the same vocabulary, not an independent filter.

Threshold rule used in the paper ("equal budget"): keep the n highest-scoring
CVEs, where n = number of CVEs the keyword filter keeps (optionally scaled, for
the 0.5x / 2x sensitivity sweep). The threshold is therefore chosen by count,
and the two filters are compared at the same retention budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25L, BM25Okapi, BM25Plus

from lexicon import LLM_QUERY_TERMS

_BM25_VARIANTS = {"okapi": BM25Okapi, "l": BM25L, "plus": BM25Plus}
_VARIANT_DEFAULT_DELTA = {"l": 0.5, "plus": 1.0}


def fit_bm25(tokenized_corpus: list[list[str]], variant: str = "okapi",
             k1: float = 1.5, b: float = 0.75, delta: float | None = None):
    """Fit a BM25 model over the tokenized corpus (computes IDF + avgdl)."""
    variant = variant.lower()
    if variant not in _BM25_VARIANTS:
        raise ValueError(f"variant must be one of {list(_BM25_VARIANTS)}, got {variant!r}")
    cls = _BM25_VARIANTS[variant]
    if variant == "okapi":
        return cls(tokenized_corpus, k1=k1, b=b)
    d = _VARIANT_DEFAULT_DELTA[variant] if delta is None else delta
    return cls(tokenized_corpus, k1=k1, b=b, delta=d)


def score_corpus(bm25, query_terms: list[str] | None = None) -> np.ndarray:
    """BM25 relevance score of every corpus document against the LLM query."""
    q = LLM_QUERY_TERMS if query_terms is None else query_terms
    return np.asarray(bm25.get_scores(q), dtype=float)


def budget_threshold(scores: np.ndarray, n_keep: int) -> float:
    """Score of the n_keep-th highest document. `scores >= threshold` keeps
    n_keep documents, or slightly more if several documents tie at the threshold."""
    scores = np.asarray(scores, float)
    if n_keep <= 0:
        return float(np.max(scores) + 1)
    return float(np.sort(scores)[::-1][min(n_keep, len(scores)) - 1])


def sweep_thresholds(scores: np.ndarray, regex_labels: np.ndarray,
                     n_steps: int = 60) -> pd.DataFrame:
    """For a grid of thresholds: BM25-positive count and agreement with the
    keyword filter (regex-relative 'precision'/'recall' — NOT against truth)."""
    scores = np.asarray(scores, float)
    reg = np.asarray(regex_labels, bool)
    grid = np.linspace(float(np.min(scores)), float(np.max(scores)), n_steps)
    rows = []
    n = len(scores)
    for thr in grid:
        pred = scores >= thr
        tp = int(np.sum(pred & reg))
        fp = int(np.sum(pred & ~reg))
        fn = int(np.sum(~pred & reg))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else np.nan)
        rows.append({"threshold": thr, "bm25_pos": int(np.sum(pred)),
                     "bm25_pos_rate": np.sum(pred) / n, "agreement": float(np.mean(pred == reg)),
                     "precision_vs_regex": prec, "recall_vs_regex": rec, "f1_vs_regex": f1})
    return pd.DataFrame(rows)


def suggest_threshold(scores: np.ndarray, regex_labels: np.ndarray,
                      strategy: str = "match_regex_rate") -> float:
    """match_regex_rate (paper default) | max_f1 | percentile95."""
    scores = np.asarray(scores, float)
    reg = np.asarray(regex_labels, bool)
    if strategy == "match_regex_rate":
        return budget_threshold(scores, int(np.sum(reg)))
    if strategy == "max_f1":
        sw = sweep_thresholds(scores, reg).dropna(subset=["f1_vs_regex"])
        return float(sw.loc[sw["f1_vs_regex"].idxmax(), "threshold"]) if not sw.empty else float(np.median(scores))
    if strategy == "percentile95":
        return float(np.percentile(scores, 95))
    raise ValueError(f"unknown strategy: {strategy}")
