"""Shared parameters of the saturation analysis (steps 02-05). Change them here only."""
import pandas as pd

T0 = pd.Timestamp('2020-01-01')           # time origin for regressions and fits
DATA_DATE = pd.Timestamp('2026-07-14')    # Epoch AI Benchmarking Hub export retrieved on this date
SAT_CUTOFF = pd.Timestamp('2030-01-01')   # lifespan sample: realized/projected 95% crossing before 2030
MIN_MODELS = 8                            # lifespan sample: at least eight distinct scored models
COHORTS = [(2019, 2023), (2023, 2024), (2024, 2025), (2025, 2027)]  # [lo, hi) by first-score year
TABLE4_BENCHMARKS = ['hella_swag', 'arc_ai2', 'gsm8k', 'mmlu', 'otis_mock_aime_2024_2025', 'gpqa_diamond',
                     'math_level_5', 'cybench', 'frontiermath', 'frontiermath_tier_4', 'arc_agi_2',
                     'critpt', 'arc_agi', 'hle', 'swe_bench_verified', 'gdpval', 'fictionlivebench', 'terminalbench',
                     'exploitbench', 'osworld_2']
