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
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi, BM25L, BM25Plus


# ─────────────────────────────────────────────────────────────────────────────
# 1. Regex reference classifier
#    Mirrored VERBATIM from notebook cell a0db16a3 so this module is
#    self-contained (runs on a bare GCP VM with no notebook). KEEP IN SYNC with
#    the notebook if the lexicon changes there.
# ─────────────────────────────────────────────────────────────────────────────
TIER_1_PATTERNS = [
    r"\blarge language model", r"\bllms?\b", r"\bgenerative ai\b", r"\bgenai\b",
    r"\bchatgpt\b", r"\bopenai\b", r"\banthropic\b",
    r"\bllama[\s-]?(2|3|4|cpp|index|file)", r"\bllama[- ]?guard\b",
    r"\bgpt-?(3\.5|4|4o|5)",
    r"\b(google )?gemini (1|1\.5|2|pro|flash|ultra|nano|model|ai|llm)",
    r"\banthropic claude\b", r"\bclaude (3|3\.5|haiku|sonnet|opus|instant|2|1)",
    r"\bmistral (7b|small|medium|large|nemo)", r"\bmixtral\b",
    r"\bqwen ?\d", r"\bdeepseek\b", r"\bgrok\b", r"\bcohere (command|embed)",
    r"\bprompt injection\b", r"\bjailbreak(ing)?\b",
    r"\bsystem prompt (leak|disclosure|injection)", r"\bprompt leakage\b",
    r"\brag (system|pipeline|application)", r"\bretrieval[- ]augmented\b",
    r"\bhallucinat", r"\bfine[- ]?tun", r"\bhugging\s?face\b", r"\btransformer model",
    r"\blangchain\b", r"\bllamaindex\b", r"\bllama[- ]index\b",
    r"\bvllm\b", r"\bollama\b", r"\bbentoml\b", r"\btriton inference\b",
    r"\bmlflow\b", r"\bkubeflow\b", r"\bautogpt\b",
    r"\bgradio\b", r"\btext[- ]generation[- ]webui\b",
    r"\bvector (db|database|store)\b", r"\bembedding model\b",
    r"\bopen[- ]?webui\b", r"\bcomfyui\b", r"\binvoke[- ]?ai\b",
    r"\banything[- ]?llm\b", r"\bflowise\b", r"\bdify\b",
    r"\bhaystack\b.*\b(llm|ai|nlp)", r"\bnvidia nemo\b", r"\bguardrails (ai|llm)",
    r"\bllamafile\b", r"\btext-generation-inference\b", r"\bopenai[- ]?api\b",
    r"\binference server\b",
    r"\bchromadb\b", r"\bpinecone\b", r"\bweaviate\b", r"\bmilvus\b",
    r"\bqdrant\b", r"\bfaiss\b",
    r"\blunary\b", r"\bgiskard\b", r"\bvanna\b", r"\binstructlab\b",
    r"\bnemo[- ]guardrails\b", r"\bsafetensors\b", r"\bpickle.*model\b",
    r"\bdspy\b", r"\blitellm\b", r"\bcrewai\b",
    r"\bautogen\b.*\b(ai|llm|model)", r"\bmem ?gpt\b",
    r"\bopen[- ]assistant\b", r"\bjan[- ]ai\b", r"\blm studio\b", r"\bkoboldcpp\b",
]
_T1 = re.compile("|".join(TIER_1_PATTERNS), re.I)

TIER_2_FRAMEWORKS = [
    r"\bpytorch\b", r"\btensorflow\b", r"\bkeras\b", r"\bscikit[- ]learn\b",
    r"\bonnx\b", r"\btransformers\b", r"\bdatasets\b",
    r"\bxgboost\b", r"\blightgbm\b", r"\bopenvino\b", r"\bmlx\b",
]
_T2 = re.compile("|".join(TIER_2_FRAMEWORKS), re.I)
_T2_CONTEXT = re.compile(
    r"\b(model|\bai\b|ml\b|machine learning|deep learning|neural network|"
    r"inference|training|llm|nlp|tensor)\b", re.I)

DENY_PRODUCTS = [
    r"resi[- ]gemini", r"gemini[- ]net",
    r"crowdstrike falcon", r"falcon logscale", r"falconpro",
    r"\bphi[- ]node\b",
]
_DENY = re.compile("|".join(DENY_PRODUCTS), re.I)


def regex_relevance(text: str) -> str:
    """Notebook's `classify_relevance`, verbatim. Returns 'adversarial',
    'supply_chain', or '' (not LLM-relevant)."""
    if _DENY.search(text):
        return ""
    if _T1.search(text):
        return "adversarial"
    if _T2.search(text) and _T2_CONTEXT.search(text):
        return "supply_chain"
    return ""


def regex_is_relevant(text: str) -> bool:
    """Binary view of the regex classifier (this module's scope is binary)."""
    return regex_relevance(text) != ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenization
#    BM25 works on bags of tokens. We first collapse a handful of multi-word key
#    phrases into single underscore-joined tokens so each phrase gets its OWN idf
#    weight — otherwise "prompt injection" would be split into the common words
#    "prompt" and "injection", diluting its signal. Then we lowercase and split
#    on runs of [a-z0-9].
# ─────────────────────────────────────────────────────────────────────────────
# Multi-word phrases → single tokens. Longest first so they match before their
# substrings. Derived from the discriminative phrases in the regex lexicon.
KEY_PHRASES = [
    "large language model", "retrieval augmented", "retrieval-augmented",
    "prompt injection", "prompt leakage", "system prompt", "generative ai",
    "machine learning", "deep learning", "neural network", "hugging face",
    "vector database", "vector store", "embedding model", "transformer model",
    "inference server", "inference engine", "fine tune", "fine-tune", "fine-tuning",
    "open webui", "text generation webui", "lm studio",
]
_PHRASE_SUBS = [
    (re.compile(re.escape(p), re.I), p.replace("-", " ").replace(" ", "_"))
    for p in sorted(KEY_PHRASES, key=len, reverse=True)
]
_TOKEN_RX = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, fold key phrases to single tokens, split into alphanumeric
    tokens. Returns a list of tokens (BM25's unit of matching)."""
    if not text:
        return []
    t = text.lower()
    for rx, repl in _PHRASE_SUBS:
        t = rx.sub(repl, t)
    return _TOKEN_RX.findall(t)


# ─────────────────────────────────────────────────────────────────────────────
# 3. The query: our LLM lexicon as BM25 query terms.
#    Plain tokens/phrases (regex metacharacters stripped), tokenized the SAME way
#    as the corpus so phrase-folding lines up. "Mirrors cell a0db16a3 — keep in
#    sync": if you add a keyword to the regex, add its plain form here too.
# ─────────────────────────────────────────────────────────────────────────────
_QUERY_SOURCE_TERMS = [
    # LLM / GenAI core
    "large language model", "llm", "llms", "generative ai", "genai",
    "chatgpt", "openai", "anthropic", "claude", "gpt", "gemini",
    "llama", "llama cpp", "llama guard", "mistral", "mixtral", "qwen",
    "deepseek", "grok", "cohere",
    # attacks / LLM concepts
    "prompt injection", "jailbreak", "jailbreaking", "system prompt",
    "prompt leakage", "rag", "retrieval augmented", "hallucination",
    "fine tune", "hugging face", "transformer model", "embedding model",
    # LLM tooling / frameworks
    "langchain", "llamaindex", "vllm", "ollama", "bentoml", "triton",
    "mlflow", "kubeflow", "autogpt", "gradio", "text generation webui",
    "open webui", "comfyui", "invokeai", "anythingllm", "flowise", "dify",
    "haystack", "nemo", "guardrails", "llamafile", "litellm", "dspy",
    "crewai", "autogen", "memgpt", "koboldcpp", "jan ai", "lm studio",
    "inference server", "safetensors",
    # vector databases
    "vector database", "vector store", "chromadb", "pinecone", "weaviate",
    "milvus", "qdrant", "faiss",
    # tier-2 ML frameworks
    "pytorch", "tensorflow", "keras", "scikit learn", "onnx", "transformers",
    "xgboost", "lightgbm", "openvino", "mlx", "machine learning",
    "deep learning", "neural network",
]
# Flatten through the SAME tokenizer so phrases fold to the same tokens as docs.
LLM_QUERY_TERMS = sorted({tok for term in _QUERY_SOURCE_TERMS for tok in tokenize(term)})
_QUERY_SET = frozenset(LLM_QUERY_TERMS)

# A curated subset of *unambiguously* LLM/GenAI terms. A CVE whose only query-term
# hits are outside this set (e.g. just "ai", "model", "training", "inference")
# is likely generic ML noise rather than a genuinely LLM-relevant CVE; one that
# contains a high-signal term is a much stronger candidate. Used to TRIAGE the
# BM25-vs-regex disagreements for human review.
HIGH_SIGNAL_TERMS = frozenset({
    # model families / products
    "chatgpt", "gpt", "claude", "gemini", "llama", "mistral", "mixtral", "qwen",
    "deepseek", "grok", "cohere", "openai", "anthropic",
    # LLM concepts / attacks
    "llm", "large_language_model", "generative_ai", "genai", "prompt_injection",
    "jailbreak", "prompt_leakage", "system_prompt", "rag", "retrieval_augmented",
    "hallucination",
    # LLM tooling / serving
    "langchain", "llamaindex", "vllm", "ollama", "llamafile", "litellm", "dspy",
    "crewai", "autogpt", "autogen", "memgpt", "koboldcpp", "flowise", "dify",
    "anythingllm", "comfyui", "guardrails", "safetensors",
    # vector databases
    "chromadb", "pinecone", "weaviate", "milvus", "qdrant", "faiss",
})


def matched_query_terms(tokens) -> list[str]:
    """Which LLM query terms actually appear in a document's tokens — i.e. WHY
    BM25 gave it any score. Great for eyeballing disagreements."""
    return sorted(set(tokens) & _QUERY_SET)


def has_high_signal(tokens) -> bool:
    """True if the document contains an unambiguously-LLM term (see
    HIGH_SIGNAL_TERMS) rather than only generic ML vocabulary."""
    return bool(set(tokens) & HIGH_SIGNAL_TERMS)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BM25 fit + score
# ─────────────────────────────────────────────────────────────────────────────
_BM25_VARIANTS = {"okapi": BM25Okapi, "l": BM25L, "plus": BM25Plus}
# The delta lower-bound each variant applies (Okapi has none).
_VARIANT_DEFAULT_DELTA = {"l": 0.5, "plus": 1.0}


def fit_bm25(tokenized_corpus: list[list[str]], variant: str = "okapi",
             k1: float = 1.5, b: float = 0.75, delta: float | None = None):
    """Fit a BM25 model over the tokenized corpus (computes IDF + avgdl).

    variant:
      "okapi" — plain Okapi BM25 (default). No term-frequency lower-bound; a
                matched term's contribution can shrink toward 0 in long docs.
      "l"     — BM25L: adds a delta (default 0.5) inside the length-normalized
                TF so long-document term contributions don't vanish.
      "plus"  — BM25+: adds a delta (default 1.0) directly to the normalized TF,
                so every matched term contributes at least IDF*delta regardless
                of document length.
    delta: override the variant's default lower-bound (ignored for "okapi").
    """
    variant = variant.lower()
    if variant not in _BM25_VARIANTS:
        raise ValueError(f"variant must be one of {list(_BM25_VARIANTS)}, got {variant!r}")
    cls = _BM25_VARIANTS[variant]
    if variant == "okapi":
        return cls(tokenized_corpus, k1=k1, b=b)
    d = _VARIANT_DEFAULT_DELTA[variant] if delta is None else delta
    return cls(tokenized_corpus, k1=k1, b=b, delta=d)


def score_corpus(bm25: BM25Okapi, query_terms: list[str] | None = None) -> np.ndarray:
    """BM25 relevance score of every corpus document against the LLM query."""
    q = LLM_QUERY_TERMS if query_terms is None else query_terms
    return np.asarray(bm25.get_scores(q), dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threshold selection + regex comparison
#    We have NO ground-truth labels, so we treat the regex as a REFERENCE (not
#    truth) and report agreement plus regex-relative precision/recall. The real
#    signal is in the disagreements.
# ─────────────────────────────────────────────────────────────────────────────
def sweep_thresholds(scores: np.ndarray, regex_labels: np.ndarray,
                     n_steps: int = 60) -> pd.DataFrame:
    """For a grid of thresholds, report BM25-positive count, agreement with the
    regex, and regex-relative precision/recall/F1. `regex_labels` is a boolean
    array (True = regex says relevant)."""
    scores = np.asarray(scores, float)
    reg = np.asarray(regex_labels, bool)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    grid = np.linspace(lo, hi, n_steps)
    rows = []
    n = len(scores)
    for thr in grid:
        pred = scores >= thr
        tp = int(np.sum(pred & reg))
        fp = int(np.sum(pred & ~reg))
        fn = int(np.sum(~pred & reg))
        agree = float(np.mean(pred == reg))
        prec = tp / (tp + fp) if (tp + fp) else np.nan   # vs regex, not truth
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else np.nan)
        rows.append({"threshold": thr, "bm25_pos": int(np.sum(pred)),
                     "bm25_pos_rate": np.sum(pred) / n, "agreement": agree,
                     "precision_vs_regex": prec, "recall_vs_regex": rec, "f1_vs_regex": f1})
    return pd.DataFrame(rows)


def suggest_threshold(scores: np.ndarray, regex_labels: np.ndarray,
                      strategy: str = "match_regex_rate") -> float:
    """Pick a default threshold.
      match_regex_rate : threshold so #BM25-positive == #regex-positive (the
                         fairest same-budget comparison — recommended default).
      max_f1           : threshold maximising F1 vs the regex reference.
      percentile95     : keep the top 5% highest-scoring CVEs.
    """
    scores = np.asarray(scores, float)
    reg = np.asarray(regex_labels, bool)
    if strategy == "match_regex_rate":
        n_pos = int(np.sum(reg))
        if n_pos == 0:
            return float(np.max(scores) + 1)
        # threshold = the n_pos-th highest score
        return float(np.sort(scores)[::-1][min(n_pos, len(scores)) - 1])
    if strategy == "max_f1":
        sw = sweep_thresholds(scores, reg)
        sw = sw.dropna(subset=["f1_vs_regex"])
        return float(sw.loc[sw["f1_vs_regex"].idxmax(), "threshold"]) if not sw.empty else float(np.median(scores))
    if strategy == "percentile95":
        return float(np.percentile(scores, 95))
    raise ValueError(f"unknown strategy: {strategy}")


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
