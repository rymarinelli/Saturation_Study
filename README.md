# Benchmark Saturation

Analysis of AI benchmark saturation using data from the Epoch AI Benchmarking Hub
(https://epoch.ai/benchmarks, CC-BY 4.0, retrieved 14 July 2026) supplemented with
scores hand-collected from Anthropic and OpenAI model/system cards.

## Contents
- `code/` — numbered pipeline scripts (run in order)
- `data/` — raw long-format dataset and all derived CSVs
- `figures/` — all generated figures (PNG)
- `notebooks/` — exploratory / analysis notebooks (Epoch FDA analysis; LLM-CVE + model-card pipeline)
- `paper/` — LaTeX section with tables (formal version), Markdown drafts, and tables

## CVE analysis (paper §4.1): reproducible pipeline
Every CVE number, table and figure in the paper is produced by `code/cve/`
(`python code/cve/run_all.py`) from the NVD snapshot pinned in `data/cve/SNAPSHOT.txt`.
Outputs go to `data/cve/*.csv`, `figures/cve/` and `paper/generated/`. See
[`code/cve/README.md`](code/cve/README.md) for the data source, filter definitions, statistics and runtime.

## Model-card & LLM-CVE component (exploratory)
Beyond the Epoch pipeline, `notebooks/llm_cve_dynamics.ipynb` covers the model-card and
LLM-CVE side of the study:
- Filters NVD CVEs to LLM-relevant ones, maps them to the OWASP Top 10 for LLM
  Applications, and analyzes CVSS / volume trends against a non-LLM counterfactual baseline.
- Builds the model-card cyber-capability progression and the benchmark-saturation figure
  (Cybench, CyberGym, Cyber Range) from `data/model_card_cyber_evals.csv`.

The CVE half needs the NVD JSON sparse-clone (setup documented inside the notebook); the
model-card half runs from the CSV alone. `code/bm25_relevance.py` is a standalone module
implementing a BM25 relevance scorer as an alternative to the regex CVE filter, with a CLI
and regex-vs-BM25 / BM25-variant comparison utilities. Extra dependencies for this
component (on top of the pipeline's): `seaborn rank_bm25`.

## Pipeline
```
pip install pandas numpy scipy matplotlib scikit-fda   # Python >= 3.14: also pip install multimethod==1.12
python code/01_load_data.py data/raw_epoch data/all_benchmarks_long.csv
python code/02_saturation_analysis.py data/all_benchmarks_long.csv data/
python code/03_regressions_and_tables.py data/     # Tables 1-4 -> data/table_*.csv, data/results_master.csv, paper/results_tables.md
python code/04_fda_analysis.py data/               # FPCA details -> data/fpca_scores.csv
python code/05_figures.py data/ figures/
```

## Key definitions
- SOTA frontier: running max of reported scores by model release date; 95% crossings read on the raw date axis, curve fits on the record-setting scores resampled month-start and forward-filled
- Saturation: SOTA >= 95% of maximum attainable score
- Lifespan: first frontier score -> 95% crossing (empirical where observed, fit-implied otherwise)
- Lifespan sample: >= 8 distinct scored models and a realized or projected 95% crossing before 2030 (both set once in `code/sat_config.py`, shared by tables and figures)
- Fits: 3-parameter logistic and Gompertz, free ceiling bounded [0.9*max_observed, 1.0]
- FPCA: landmark registration at interpolated 50% crossing; cubic B-splines (7 basis fns) on [-12,+9] months
- Peak velocity: max month-over-month gain of smoothed SOTA; benchmarks past 70% only

## Data files
- raw_epoch/ — the Epoch AI Benchmarking Hub bulk export (CC-BY 4.0, retrieved 14 July 2026), unmodified
- all_benchmarks_long.csv — tidy (model, date, score, benchmark), 54 benchmarks, 3,426 rows (271 exact duplicate rows in the Epoch export removed)
- sota_monthly_panel.csv — monthly SOTA per benchmark (modeling-ready)
- sota_aligned_at_50pct.csv — panel re-anchored at 50% crossing (35 benchmarks)
- saturation_summary.csv — per-benchmark: n, SOTA, 90%/95% crossings, projections
- saturation_curve_fits.csv — logistic + Gompertz parameters and RMSE per benchmark
- results_master.csv — merged per-benchmark results (summary, best fit, peak velocity, lifespan-sample flag); basis for paper Table 4
- table_cohorts.csv, table_regressions.csv, table_fpca.csv, table_status.csv — paper Tables 1–4 (also rendered to paper/results_tables.md)
- model_card_cyber_evals.csv — cyber-eval scores (Cybench, CyberGym, Cyber Range, OpenAI Preparedness cyber-risk levels, CTF suites) hand-collected from Anthropic and OpenAI model/system cards; feeds the saturation figure's model-card panels (each row cites its `card_url`)

## Known caveats
- "First score" may postdate true benchmark release for pre-Hub benchmarks
- Developer-reported and standardized-harness scores are mixed within benchmarks
- Free-ceiling fits mid-curve underestimate ceilings (FrontierMath, GDPval, CritPT)
- Recent cohorts are right-censored; survival-analytic treatment is the planned extension
