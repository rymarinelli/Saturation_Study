# Benchmark Saturation

Analysis of AI benchmark saturation using data from the Epoch AI Benchmarking Hub
(https://epoch.ai/benchmarks, CC-BY 4.0, retrieved 14 July 2026) supplemented with
scores hand-collected from Anthropic and OpenAI model/system cards.

## Contents
- `code/` — numbered pipeline scripts (run in order)
- `data/` — raw long-format dataset and all derived CSVs
- `figures/` — all generated figures (PNG)
- `paper/` — LaTeX section with tables (formal version), Markdown drafts, and tables

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

## Known caveats
- "First score" may postdate true benchmark release for pre-Hub benchmarks
- Developer-reported and standardized-harness scores are mixed within benchmarks
- Free-ceiling fits mid-curve underestimate ceilings (FrontierMath, GDPval, CritPT)
- Recent cohorts are right-censored; survival-analytic treatment is the planned extension
