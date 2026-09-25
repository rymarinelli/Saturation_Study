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

## Model-card & LLM-CVE component
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
pip install pandas numpy scipy matplotlib scikit-fda
python code/01_load_data.py data/raw_epoch data/all_benchmarks_long.csv
python code/02_saturation_analysis.py data/all_benchmarks_long.csv data/
python code/03_regressions_and_tables.py data/     # prints Table 1-2 statistics
python code/04_fda_analysis.py data/               # FPCA (Table: FPCA components)
python code/05_figures.py data/ figures/
```

## Key definitions
- SOTA frontier: running max of reported scores by model release date, monthly forward-filled
- Saturation: SOTA >= 95% of maximum attainable score
- Lifespan: first frontier score -> 95% crossing (empirical where observed, fit-implied otherwise)
- Fits: 3-parameter logistic and Gompertz, free ceiling bounded [0.9*max_observed, 1.0]
- FPCA: landmark registration at interpolated 50% crossing; cubic B-splines (7 basis fns) on [-12,+9] months
- Peak velocity: max month-over-month gain of smoothed SOTA; benchmarks past 70% only

## Data files
- raw_epoch/ — the Epoch AI Benchmarking Hub bulk export (CC-BY 4.0, retrieved 14 July 2026), unmodified
- all_benchmarks_long.csv — tidy (model, date, score, benchmark), 54 benchmarks, ~3,400 rows
- sota_monthly_panel.csv — monthly SOTA per benchmark (modeling-ready)
- sota_aligned_at_50pct.csv — panel re-anchored at 50% crossing (35 benchmarks)
- saturation_summary.csv — per-benchmark: n, SOTA, 90%/95% crossings, projections
- saturation_curve_fits.csv — logistic + Gompertz parameters and RMSE per benchmark
- results_master.csv — merged per-benchmark results table (basis for paper Table 4)
- model_card_cyber_evals.csv — cyber-eval scores (Cybench, CyberGym, Cyber Range, OpenAI Preparedness cyber-risk levels, CTF suites) hand-collected from Anthropic and OpenAI model/system cards; feeds the saturation figure's model-card panels (each row cites its `card_url`)

## Known caveats
- "First score" may postdate true benchmark release for pre-Hub benchmarks
- Developer-reported and standardized-harness scores are mixed within benchmarks
- Free-ceiling fits mid-curve underestimate ceilings (FrontierMath, GDPval, CritPT)
- Recent cohorts are right-censored; survival-analytic treatment is the planned extension
