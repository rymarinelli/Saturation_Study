"""
bm25_relevance.py — a BM25 relevance classifier for LLM-related CVEs.
====================================================================

WHY THIS EXISTS
---------------
The notebook decides "is this CVE about an LLM system?" with a hand-tuned regex
(`classify_relevance`, notebook cell a0db16a3). That is binary and brittle: a
description either contains one of the keyword patterns or it does not. This
module builds a *second opinion* on the same decision using **BM25**, runs it
**alongside** the regex, and reports where the two agree and disagree. It does
NOT replace the regex — the disagreements are the point (they are the candidate
list for human adjudication / classifier audit).

WHAT BM25 IS (read this — it's the thing to learn)
--------------------------------------------------
BM25 ("Best Matching 25") is the classic information-retrieval ranking function
behind search engines. It scores how well a *document* D matches a *query* Q.
Here:
    document D = a CVE's text (English description + CPE strings)
    query    Q = our LLM lexicon (LLM_QUERY_TERMS below)

    score(D, Q) = sum over each query term q of:

                                 tf(q, D) * (k1 + 1)
        IDF(q) * -----------------------------------------------------
                 tf(q, D) + k1 * (1 - b + b * |D| / avgdl)

Three ideas do all the work:

1. tf(q, D) — how many times term q appears in document D. More is better, but
   with SATURATION: k1 (~1.5) makes the 1st occurrence matter a lot and the 5th
   barely add anything (a CVE that says "LLM" once is already clearly about LLMs).

2. IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)   [N = corpus size,
   df(q) = # docs containing q]. This is INVERSE DOCUMENT FREQUENCY: a term that
   occurs in few CVEs (e.g. "vllm", "langchain") is highly discriminative and gets
   a big weight; a term in almost every CVE (e.g. "server", "model") gets almost
   none. This is why BM25 needs the WHOLE corpus first — it has to count df and N.

3. |D| / avgdl — LENGTH NORMALIZATION (strength set by b ~ 0.75). A short CVE that
   is mostly about an LLM scores higher than a long CVE that mentions an LLM once
   in passing, because the long one's score is divided down by its length.

Sum those contributions over all query terms → one relevance score per CVE.
Threshold the score → a binary keep/discard decision we can compare to the regex.

IMPORTANT CAVEAT — this is NOT semantic matching. Because our query is literally
the keyword list, BM25 here behaves like a *graded, rarity-weighted,
length-normalized version of the regex*. Its added value is (a) a continuous
score you can rank and threshold, and (b) exposing borderline CVEs the binary
regex silently drops or over-keeps. It will NOT catch an LLM CVE that shares no
vocabulary with the query (a paraphrase). Catching those needs embeddings /
semantic search — a heavier, possible later step.

USAGE
-----
As a library (e.g. from the notebook):
    from bm25_relevance import build_comparison
    df = build_comparison(nvd_dir="llm_cve_analysis/nvd",
                          out_csv="llm_cve_analysis/bm25_comparison.csv")

As a script (e.g. on a GCP VM over the full ~170k-CVE corpus):
    python bm25_relevance.py --nvd llm_cve_analysis/nvd \\
                             --out llm_cve_analysis/bm25_comparison.csv

GCP recipe: spin up a small VM, clone the NVD sparse-mirror the same way the
notebook does (git sparse-checkout of CVE-2022..CVE-2026), `pip install
rank_bm25`, run the command above, copy the CSV back. ~170k short docs fit in a
few hundred MB of RAM and score in well under a minute; no cluster needed. If you
ever want it faster, `bm25s` is a drop-in faster BM25 implementation.

Dependencies: rank_bm25, numpy, pandas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The lexicon, tokenizer and BM25 helpers now live in code/cve/ (the paper's
# reproducible pipeline, entry point code/cve/run_all.py). This module keeps the
# exploratory walk/compare utilities used by notebooks/llm_cve_dynamics.ipynb.
sys.path.insert(0, str(Path(__file__).resolve().parent / "cve"))
from lexicon import (DENY_PRODUCTS, HIGH_SIGNAL_TERMS, KEY_PHRASES,  # noqa: E402,F401
                     LLM_QUERY_TERMS, TIER_1_PATTERNS, TIER_2_FRAMEWORKS,
                     has_high_signal, keyword_relevance, matched_query_terms, tokenize)
from bm25 import (fit_bm25, score_corpus, suggest_threshold,  # noqa: E402,F401
                  sweep_thresholds)

regex_relevance = keyword_relevance


def regex_is_relevant(text: str) -> bool:
    return regex_relevance(text) != ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. NVD walk + comparison builder
#    Reuses the scan structure from notebook cell 7d125f41. The notebook's
#    lightweight df_non dropped descriptions, so this module owns corpus
#    construction and reads the NVD JSON directly.
# ─────────────────────────────────────────────────────────────────────────────
def _extract_combined(cve: dict) -> str | None:
    """English description + CPE match strings — identical to the notebook's
    `combined`. Returns None if there is no English description."""
    desc = next((d["value"] for d in cve.get("descriptions", [])
                 if d.get("lang") == "en"), "")
    if not desc:
        return None
    cpe_texts = []
    for cfg in cve.get("configurations", []):
        for node in cfg.get("nodes", []):
            for m in node.get("cpeMatch", []):
                cpe_texts.append(m.get("criteria", ""))
    return f"{desc} {' '.join(cpe_texts)}"


def walk_nvd(nvd_dir: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Walk the sparse-cloned NVD tree (CVE-2*/**/*.json) and return one row per
    CVE with: cve_id, published, combined (text), tokens, regex_relevance,
    regex_is_relevant. `limit` caps the number of CVEs (for quick tests)."""
    nvd = Path(nvd_dir)
    if not nvd.exists():
        raise FileNotFoundError(f"NVD directory not found: {nvd}")
    rows = []
    for year_dir in sorted(nvd.glob("CVE-2*")):
        for f in year_dir.rglob("*.json"):
            try:
                cve = json.loads(f.read_text())
            except Exception:
                continue
            combined = _extract_combined(cve)
            if combined is None:
                continue
            rel = regex_relevance(combined)
            rows.append({
                "cve_id": cve.get("id"),
                "published": cve.get("published"),
                "combined": combined,
                "tokens": tokenize(combined),
                "regex_relevance": rel,
                "regex_is_relevant": rel != "",
            })
            if limit is not None and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_comparison(nvd_dir: str | Path, out_csv: str | Path | None = None,
                     threshold_strategy: str = "match_regex_rate",
                     variant: str = "okapi", delta: float | None = None,
                     limit: int | None = None) -> pd.DataFrame:
    """End-to-end: walk NVD → fit BM25 → score every CVE → attach the regex
    reference label and a BM25 label at the chosen threshold. `variant` selects
    okapi / l / plus (see fit_bm25). Writes `out_csv` (minus the bulky
    token/text columns) if given, and returns the full frame."""
    df = walk_nvd(nvd_dir, limit=limit)
    if df.empty:
        raise RuntimeError("No CVEs with English descriptions found under nvd_dir.")
    bm25 = fit_bm25(df["tokens"].tolist(), variant=variant, delta=delta)
    df["bm25_score"] = score_corpus(bm25)
    thr = suggest_threshold(df["bm25_score"].to_numpy(),
                            df["regex_is_relevant"].to_numpy(), threshold_strategy)
    df["bm25_is_relevant"] = df["bm25_score"] >= thr
    df["desc_snippet"] = df["combined"].str.slice(0, 240)
    df["doc_len"] = df["tokens"].map(len)
    df["matched_terms"] = df["tokens"].map(lambda ts: "|".join(matched_query_terms(ts)))
    df["high_signal"] = df["tokens"].map(has_high_signal)
    df.attrs["bm25_threshold"] = thr
    df.attrs["threshold_strategy"] = threshold_strategy
    df.attrs["variant"] = variant

    n = len(df)
    agree = float(np.mean(df["bm25_is_relevant"] == df["regex_is_relevant"]))
    bm25_only = int(np.sum(df["bm25_is_relevant"] & ~df["regex_is_relevant"]))
    regex_only = int(np.sum(~df["bm25_is_relevant"] & df["regex_is_relevant"]))
    print(f"Scored {n:,} CVEs | variant={variant} | threshold={thr:.3f} ({threshold_strategy})")
    print(f"  regex-relevant: {int(df['regex_is_relevant'].sum()):,}  "
          f"BM25-relevant: {int(df['bm25_is_relevant'].sum()):,}")
    print(f"  agreement: {agree*100:.2f}%  |  BM25-only: {bm25_only:,}  "
          f"regex-only: {regex_only:,}")

    if out_csv is not None:
        cols = ["cve_id", "published", "regex_relevance", "regex_is_relevant",
                "bm25_score", "bm25_is_relevant", "doc_len", "matched_terms",
                "high_signal", "desc_snippet"]
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_csv(out, index=False)
        print(f"  wrote {out}")
    return df


def compare_variants(nvd_dir: str | Path,
                     variants: tuple[str, ...] = ("okapi", "l", "plus"),
                     threshold_strategy: str = "match_regex_rate",
                     delta: float | None = None,
                     limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk NVD ONCE, then score the same corpus under each BM25 variant and
    compare how the positive set (and its disagreements with the regex) shifts.

    For each variant we report, at the `match_regex_rate` threshold (equal keep
    budget, so counts are comparable): agreement with the regex, the BM25-only
    (candidate regex-miss) and regex-only (candidate false-positive) counts, the
    positive-set Jaccard overlap, and — to test whether BM25+ pulls in longer
    documents — the mean token length of the BM25-only set vs the whole corpus.

    Returns (summary_df, per_cve_df). The per-CVE frame carries score_<v> and
    pos_<v> columns for every variant for deeper inspection.
    """
    df = walk_nvd(nvd_dir, limit=limit)
    if df.empty:
        raise RuntimeError("No CVEs with English descriptions found under nvd_dir.")
    reg = df["regex_is_relevant"].to_numpy()
    df["doc_len"] = df["tokens"].map(len)
    corpus_mean_len = round(float(df["doc_len"].mean()), 1)

    rows = []
    for v in variants:
        bm25 = fit_bm25(df["tokens"].tolist(), variant=v, delta=delta)
        sc = score_corpus(bm25)
        thr = suggest_threshold(sc, reg, threshold_strategy)
        pred = sc >= thr
        df[f"score_{v}"] = sc
        df[f"pos_{v}"] = pred
        bm25_only = pred & ~reg
        inter = int(np.sum(pred & reg))
        union = int(np.sum(pred | reg))
        rows.append({
            "variant": v,
            "threshold": round(float(thr), 3),
            "bm25_pos": int(pred.sum()),
            "agreement_%": round(float(np.mean(pred == reg)) * 100, 3),
            "bm25_only": int(bm25_only.sum()),
            "regex_only": int(np.sum(~pred & reg)),
            "jaccard_%": round(100 * inter / max(1, union), 2),
            "bm25_only_mean_len": (round(float(df.loc[bm25_only, "doc_len"].mean()), 1)
                                   if bm25_only.any() else float("nan")),
            "corpus_mean_len": corpus_mean_len,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if "okapi" in variants and "plus" in variants:
        flip = int((df["pos_okapi"] != df["pos_plus"]).sum())
        print(f"\nCVEs whose keep/discard flips between Okapi and BM25+: {flip}")
        gained = int((df["pos_plus"] & ~df["pos_okapi"]).sum())
        lost = int((~df["pos_plus"] & df["pos_okapi"]).sum())
        print(f"  BM25+ keeps but Okapi drops: {gained}   |   Okapi keeps but BM25+ drops: {lost}")
    return summary, df


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI — for the GCP run
# ─────────────────────────────────────────────────────────────────────────────
def _main() -> None:
    ap = argparse.ArgumentParser(
        description="BM25 vs regex LLM-relevance comparison over the NVD corpus.")
    ap.add_argument("--nvd", required=True, help="Path to the sparse-cloned NVD dir (contains CVE-2* subdirs).")
    ap.add_argument("--out", default="bm25_comparison.csv", help="Output CSV path.")
    ap.add_argument("--strategy", default="match_regex_rate",
                    choices=["match_regex_rate", "max_f1", "percentile95"],
                    help="Threshold-selection strategy.")
    ap.add_argument("--variant", default="okapi", choices=["okapi", "l", "plus"],
                    help="BM25 variant: okapi (default), l (BM25L), plus (BM25+).")
    ap.add_argument("--delta", type=float, default=None,
                    help="Override the l/plus delta lower-bound (ignored for okapi).")
    ap.add_argument("--compare-variants", action="store_true",
                    help="Score under okapi/l/plus and print a comparison instead of writing one CSV.")
    ap.add_argument("--limit", type=int, default=None, help="Cap #CVEs (for a quick test run).")
    args = ap.parse_args()
    if args.compare_variants:
        compare_variants(args.nvd, threshold_strategy=args.strategy,
                         delta=args.delta, limit=args.limit)
    else:
        build_comparison(args.nvd, args.out, threshold_strategy=args.strategy,
                         variant=args.variant, delta=args.delta, limit=args.limit)


if __name__ == "__main__":
    _main()
