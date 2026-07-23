# Results tables

## Table 1. Benchmark saturation dynamics by introduction cohort

| Introduction cohort | Benchmarks (n) | Median lifespan to 95% (yr) | n | Median fitted rate k | Median peak velocity (pts/mo) | n |
|---|---|---|---|---|---|---|
| 2019–2022 | 14 | 4.7 | 3 | 0.52 | 5.9 | 10 |
| 2023      | 6  | 2.7 | 5 | 1.84 | 7.0 | 6  |
| 2024      | 22 | 2.7 | 22 | 2.68 | 10.4 | 11 |
| 2025–2026 | 11 | 2.3 | 9 | — | — | 0 |

*Notes.* Cohorts defined by date of first frontier score in the dataset. Lifespan = interval from first frontier score to the 95% crossing (empirical where observed, fit-implied otherwise); restricted to benchmarks with ≥8 scored models and a realized or projected crossing before 2030. Fitted rate k from the better of logistic/Gompertz fits by RMSE (≥12 monthly observations required). Peak velocity = maximum month-over-month gain of the smoothed SOTA series, computed only for benchmarks whose frontier has passed 70%; no 2025–2026 benchmark qualifies yet, and the fitted-k entry for this cohort rests on a single benchmark and is suppressed.

## Table 2. Regression estimates of the change in saturation speed

| Outcome | Sample | Slope (per year of introduction) | r | p | n |
|---|---|---|---|---|---|
| Lifespan to 95% (months) | Realized + projected crossings | −5.4 | −0.53 | <0.001 | 39 |
| Lifespan to 95% (months) | Empirically realized crossings only | −7.5 | −0.91 | 0.011 | 6 |
| log fitted rate k | Best sigmoid fit per benchmark | +0.30 (×1.35/yr) | 0.57 | 0.0001 | 40 |
| log peak velocity | Benchmarks past 70% | +0.09 (×1.09/yr) | 0.29 | 0.15 | 27 |
| FPCA PC1 score (steepness mode) | Fully observed registered curves | +0.11 | 0.42 | 0.35 | 7 |

*Notes.* Each row regresses the stated outcome on the benchmark's introduction date (year, continuous). Rows are ordered from strongest to weakest structural assumptions; the two nonparametric estimates (rows 4–5) are individually underpowered but concordant in sign with the parametric estimates.

## Table 3. Functional principal component analysis of registered trajectories

| Component | Variance explained | Correlation with within-window gain | Interpretation |
|---|---|---|---|
| PC1 | 71.0% | −0.87 | Steepness of the saturation trajectory |
| PC2 | 23.5% | +0.28 | Asymmetry about the 50% crossing |

*Notes.* Trajectories landmark-registered at the interpolated 50% crossing, represented in a cubic B-spline basis (7 basis functions) on [−12, +9] months, n = 7 benchmarks with full coverage of the window. The dominant mode of shape variation is steepness, supporting rate as the natural object of study independent of any sigmoid assumption.

## Table 4. Saturation status of selected benchmarks

| Benchmark | First score | Current SOTA | 95% crossed | 95% projected | Lifespan (yr) | Best fit | Ceiling | k | Peak vel. (pts/mo) |
|---|---|---|---|---|---|---|---|---|---|
| HellaSwag | 2019-11 | 95.3% | 2023-03 | — | 3.4 | logistic | 1.00 | 0.77 | 5.4 |
| GSM8K | 2020-06 | 94.5% | — | — | — | Gompertz | 0.93 | 2.74 | 18.7 |
| MMLU | 2021-08 | 88.1% | — | — | — | Gompertz | 0.86 | 2.12 | 11.4 |
| OTIS Mock AIME | 2023-03 | 100% | 2025-12 | — | 2.7 | Gompertz | 0.94 | 3.91 | 13.4 |
| GPQA Diamond | 2023-03 | 94.6% | — | 2026-07 | 3.3 | logistic | 1.00 | 1.16 | 5.4 |
| MATH Level 5 | 2023-06 | 98.1% | 2025-01 | — | 1.6 | logistic | 1.00 | 2.41 | 12.5 |
| Cybench | 2024-02 | 93.0% | — | 2026-12 | 2.8 | logistic | 1.00 | 2.50 | 13.0 |
| FrontierMath | 2024-06 | 52.4% | — | 2028-01 | 3.6 | Gompertz | 0.62 | 1.58 | — |
| FrontierMath Tier 4 | 2024-06 | 47.9% | — | 2027-07 | 3.1 | logistic | 0.65 | 3.28 | — |
| ARC-AGI-2 | 2024-07 | 92.5% | — | 2026-06 | 1.9 | logistic | 0.94 | 6.20 | 15.6 |
| CritPT | 2024-07 | 32.3% | — | 2028-03 | 3.7 | logistic | 0.38 | 3.98 | — |
| ARC-AGI | 2024-09 | 98.0% | 2026-02 | — | 1.4 | logistic | 1.00 | 2.73 | 8.8 |
| HLE | 2024-09 | 46.4% | — | 2028-01 | 3.3 | Gompertz | 0.98 | 0.97 | — |
| SWE-bench Verified | 2024-11 | 83.5% | — | 2027-06 | 2.6 | logistic | 0.78 | 5.37 | 10.4 |
| GDPval | 2024-11 | 49.7% | — | 2027-11 | 3.0 | logistic | 0.52 | 4.40 | — |
| Terminal-Bench | 2025-06 | 90.2% | — | 2026-08 | 1.1 | — | — | — | — |
| ExploitBench | 2025-10 | 41.0% | — | 2028-01 | 2.3 | — | — | — | — |
| OSWorld-2 | 2026-02 | 20.6% | — | 2027-12 | 1.8 | — | — | — | — |

*Notes.* "First score" = first frontier observation in the dataset, which for older benchmarks may postdate true release. GSM8K and MMLU never cross 95% under the free-ceiling fits (fitted ceilings 0.93 and 0.86), reflecting label-noise floors; their lifespan cells are accordingly empty. Fitted ceilings well below current plausibility (e.g., FrontierMath 0.62, GDPval 0.52, CritPT 0.38) are mid-curve artifacts and should be read as lower bounds, not estimates of true attainable performance. Dashes in fit columns indicate fewer than 12 monthly observations. Projections assume the fitted sigmoid; treat as rough extrapolations.
