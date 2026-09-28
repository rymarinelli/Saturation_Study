# CVE analysis (RQ1): reproducible pipeline

This pipeline produces every CVE number, table and figure in the paper's descriptive CVE analysis (§4.1, Tables `tab:cve-retained` / `tab:cve-robustness`, appendix `app:cve-lexicon`). It is fully rule-based and deterministic. No CVE is labelled by hand or by a language model.

## Data source and snapshot

- **Source:** NVD CVE records via the JSON mirror [fkie-cad/nvd-json-data-feeds](https://github.com/fkie-cad/nvd-json-data-feeds) (one JSON file per CVE, NVD API 2.0 schema).
- **Pinned snapshot:** see [`data/cve/SNAPSHOT.txt`](../../data/cve/SNAPSHOT.txt), which holds the commit SHA, commit date (the data cut-off) and retrieval date. [`fetch_nvd.sh`](fetch_nvd.sh) fetches exactly that commit (shallow, all `CVE-YYYY` directories, ~3.9 GB on disk).
- **Population:** records *published* from 2022-01-01 up to, but excluding, the snapshot commit day (UTC), with `vulnStatus != "Rejected"` and an English description. All ID-year directories are scanned, because CVE IDs reserved in earlier years are routinely published later. `data/cve/id_year_contributions.csv` shows how much this matters per year.

## Re-running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cve/run_all.py              # fetches the pinned snapshot if data/cve/nvd is missing
.venv/bin/python code/cve/run_all.py --nvd DIR    # reuse a clone (must be at the pinned SHA)
```

Runtime on a 10-core laptop: roughly 5 min to fetch, roughly 15–20 min for the first scan (I/O-bound; the result is cached in `data/cve/cache/`), and under 2 min for BM25 + analysis + figures on later runs.

Checks (not needed to reproduce the paper):

```bash
.venv/bin/python code/cve/verify.py stats                     # HL / P(superiority) / credible intervals vs brute force
.venv/bin/python code/cve/verify.py legacy --nvd OLD_CLONE    # port fidelity vs the exploratory notebook
bash code/cve/check_anonymity.sh                              # no identifying strings in tracked files
```

## Relevance definitions

| Name | Definition |
|---|---|
| **keyword** | Regex filter (`lexicon.keyword_relevance`) on *description + CPE strings*. Tier 1: explicit LLM/GenAI lexicon and LLM-specific tools. Tier 2: classical ML frameworks (TensorFlow, PyTorch, …), only together with an ML-context word. Deny-list for known collisions. |
| **bm25** | Okapi BM25 (k1 = 1.5, b = 0.75) of the same text against the lexicon in token form (`lexicon.LLM_QUERY_TERMS`). The *n* highest-scoring CVEs are kept, where *n* = the size of the keyword set ("equal budget"). |
| **intersection** | keyword ∩ bm25: the **primary analysis set**. |
| `*_tier1` | The same, restricted to Tier-1 keyword matches (drops ML-framework CVEs). |
| `*_0.5x`, `*_2x` | BM25 budget at 0.5× / 2× the keyword count. |

**BM25 is not an independent validator.** Its query is the keyword lexicon, and its threshold is set by count. Agreement between the two filters therefore bounds *specification* sensitivity (hard regex vs graded, rarity-weighted, length-normalized matching of the same vocabulary), not validity. No ground-truth labels exist, and none were created.

**Why Okapi BM25.** During development (on an earlier NVD snapshot), the same corpus was scored under Okapi, BM25L and BM25+ at the equal-budget threshold. BM25+ was rank-identical to Okapi: its δ adds a near-uniform baseline. BM25L diverged from the keyword set (Jaccard about 68% → 49%), and the extra CVEs it kept were about six times longer than the corpus average. Its length correction was pulling in long records that mention an LLM only in passing. Plain Okapi is therefore used.

The full pattern lists are in [`lexicon.py`](lexicon.py) and are exported verbatim to `paper/generated/cve_lexicon.tex`.

## OWASP Top 10 for LLM Applications (2025) mapping

Rule-based, in `lexicon.categorize_owasp`. A CVE gets every category whose description regex matches or whose CWE set intersects the record's CWEs. Tier-2 CVEs without a match default to LLM03. Categories with fewer than 20 CVEs are merged into "Other" in the figure, and CVEs matching no category are reported as "Unmapped".

The CWE sets follow how CVE analysts typically classify these weaknesses:

| Category | CWEs | Rationale |
|---|---|---|
| LLM02 Sensitive information disclosure | 200, 201, 209, 359, 532, 538, 540 | information-disclosure family |
| LLM03 Supply chain | 78, 94, 502, 829, 915, 1188, 1395 | deserialization, code injection, malicious or vulnerable dependencies |
| LLM05 Improper output handling | 79, 80, 601, 918 | XSS, SSRF, open redirect: typical when model output is rendered or followed |
| LLM06 Excessive agency | 269, 285, 732 | authorization bypass on tool-call surfaces |
| LLM10 Unbounded consumption | 400, 674, 770, 789, 834, 835 | resource exhaustion |

LLM01, LLM04, LLM07 and LLM09 deliberately have no CWE component: they describe model behaviors that the CWE taxonomy does not model directly, so they are matched on description patterns only.

## Statistics

- **Shares:** Beta posterior with a Jeffreys Beta(½, ½) prior; posterior mean and 95% equal-tailed credible interval. First-vs-last-year contrasts (difference, ratio, P(increase)) come from 100,000 Monte Carlo draws.
- **CVSS:** v3.x base score (3.1 preferred over 3.0; NVD "Primary" assessment preferred over CNA "Secondary"). The LLM-related set is compared with all other CVEs by the Hodges–Lehmann shift (median of all pairwise differences), with a 10,000-resample percentile bootstrap. Also reported: the Mann–Whitney U test (asymptotic, tie-corrected) and the probability of superiority P(X>Y)+½P(X=Y). Because scores lie on a 0.1 grid, all pairwise quantities are computed exactly from histograms (`stats.py`, checked against brute force in `verify.py`). Ties are heavy, so the HL estimate and its interval take grid values.
- **Sensitivity:** v4.0 scores; NVD-assessed-only and CNA-assessed-only subsets; per-year shifts; a window cut at 2026-06-30 to limit enrichment lag.
- Seeds are fixed in [`config.py`](config.py); each quantity has its own named RNG stream, so outputs don't depend on execution order.

## Outputs → paper

| File (`data/cve/`) | Used for |
|---|---|
| `population_info.csv` | window, rejected count, set sizes, BM25 thresholds |
| `retained_counts.csv` | Table `tab:cve-retained` (per-year totals, keyword / BM25 / both, Jaccard, share + CrI) |
| `filter_agreement.csv` | keyword-only / BM25-only counts, Jaccard per year and per budget |
| `share_by_year.csv`, `share_contrasts.csv` | share trend + credible intervals, all definitions |
| `share_by_quarter.csv` | Figure `cve_share_quarterly` |
| `cvss_shift.csv` | HL shift, bootstrap CI, Mann–Whitney, P(superiority); all definitions and sensitivity variants |
| `cvss_coverage.csv` | CVSS version / source / missingness per group and year |
| `owasp_counts.csv`, `owasp_by_year.csv` | Figure `cve_owasp_cvss`, per-category n |
| `robustness.csv` | Table `tab:cve-robustness` |
| `id_year_contributions.csv` | publish year × CVE-ID year |
| `analysis_set.csv` | every CVE retained by keyword or BM25, with flags, scores, OWASP categories and a text snippet |

The LaTeX fragments in `paper/generated/` (`cve_numbers.tex` macros, `cve_tables.tex`, `cve_lexicon.tex`) are generated from these CSVs by `texgen.py`. Figures go to `figures/cve/`.

## Relation to the earlier exploratory analysis

An earlier exploratory notebook (not part of this artifact) produced the AISec '26 numbers. The paper numbers come from this pipeline. Differences: rejected records are excluded, pre-2022 ID directories are scanned (publish-date binning), CVSS v4.0 and the NVD/CNA source are recorded, BM25 enters the analysis set, and uncertainty is quantified. With the notebook's settings, the scanner reproduces the notebook's kept set exactly (`verify.py legacy`).
