"""Fixed parameters of the CVE analysis. Every number in the paper's CVE section
is a deterministic function of these values and the pinned NVD snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cve"               # committed outputs (CSV)
CACHE = DATA / "cache"                     # gitignored full-corpus intermediates
NVD_DIR = DATA / "nvd"                     # gitignored snapshot clone (fetch_nvd.sh)
SNAPSHOT_FILE = DATA / "SNAPSHOT.txt"
FIG_DIR = ROOT / "figures" / "cve"
TEX_DIR = ROOT / "paper" / "generated"     # \input{} into the paper

# Analysis window: CVEs *published* in [WINDOW_START, window_end()).
WINDOW_START = pd.Timestamp("2022-01-01", tz="UTC")
# Sensitivity cut for NVD enrichment lag (CPE strings and NVD CVSS scores are
# added weeks to months after publication): drop the last, partial quarter.
LAG_CUT = pd.Timestamp("2026-07-01", tz="UTC")

SEED = 20260925
N_BOOT = 10_000          # bootstrap resamples (Hodges-Lehmann interval)
N_MC = 100_000           # Monte Carlo draws from Beta posteriors
CRED = 0.95
JEFFREYS = (0.5, 0.5)    # Beta prior for shares

BM25_VARIANT = "okapi"
BM25_K1 = 1.5
BM25_B = 0.75
BUDGET_MULTS = (0.5, 1.0, 2.0)   # BM25 retention budget relative to keyword filter

MIN_OWASP_N = 20         # OWASP categories with fewer CVEs are merged into "Other"


def read_snapshot() -> dict:
    out = {}
    for line in SNAPSHOT_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def window_end() -> pd.Timestamp:
    """Exclusive end of the window: midnight UTC of the snapshot commit day, so
    the last (possibly partially mirrored) day is not counted."""
    ts = pd.Timestamp(read_snapshot()["commit_date"]).tz_convert("UTC")
    return ts.normalize()
