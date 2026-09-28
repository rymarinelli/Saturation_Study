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
