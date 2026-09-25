"""
scan.py — one pass over the NVD snapshot -> one row per CVE record.

For every CVE-YYYY/**/CVE-*.json file we extract: ID, ID year, publish date,
vulnStatus, English description, CPE match strings, CWEs, CVSS v3.x and v4.0
base scores (NVD "Primary" score preferred over CNA "Secondary"), whether a
v2 score exists, and the keyword-filter label. Nothing is dropped here, so the
cache can also serve the legacy cross-check (verify.py); windowing and the
rejected-record exclusion happen in analyze.py.

    python code/cve/scan.py [--nvd DIR] [--dirs CVE-2022 CVE-2023 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import config as C
from lexicon import keyword_relevance


def _pick(entries: list[dict]) -> dict | None:
    """Prefer the NVD ('Primary') assessment; otherwise the first CNA one."""
    if not entries:
        return None
    for e in entries:
        if e.get("type") == "Primary":
            return e
    return entries[0]


def parse_record(path: str) -> dict | None:
    try:
        with open(path, "rb") as fh:
            cve = json.load(fh)
    except Exception:
        return None
    desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
    cpes = [m.get("criteria", "")
            for cfg in cve.get("configurations", [])
            for node in cfg.get("nodes", [])
            for m in node.get("cpeMatch", [])]
    cwes = sorted({d["value"] for w in cve.get("weaknesses", [])
                   for d in w.get("description", [])
                   if str(d.get("value", "")).startswith("CWE-")})
    metrics = cve.get("metrics", {})

    v3 = v3_ver = None
    for key, ver in (("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0")):
        e = _pick(metrics.get(key) or [])
        if e is not None:
            v3, v3_ver = e, ver
            break
    v4 = _pick(metrics.get("cvssMetricV40") or [])

    text = f"{desc} {' '.join(cpes)}" if desc else ""
    cve_id = cve.get("id", "")
    return {
        "cve_id": cve_id,
        "id_year": int(cve_id.split("-")[1]) if cve_id.count("-") >= 2 else None,
        "published": cve.get("published"),
        "vuln_status": cve.get("vulnStatus"),
        "has_en_desc": bool(desc),
        "text": text,
        "cwes": "|".join(cwes),
        "cvss3_score": v3["cvssData"].get("baseScore") if v3 else None,
        "cvss3_version": v3_ver,
        "cvss3_source": v3.get("type") if v3 else None,
        "cvss4_score": v4["cvssData"].get("baseScore") if v4 else None,
        "cvss4_source": v4.get("type") if v4 else None,
        "has_cvss2": bool(metrics.get("cvssMetricV2")),
        "keyword": keyword_relevance(text) if text else "",
    }


def list_files(nvd_dir: Path, dirs: list[str] | None = None) -> list[str]:
    year_dirs = sorted(p for p in nvd_dir.glob("CVE-*") if p.is_dir())
    if dirs:
        year_dirs = [p for p in year_dirs if p.name in set(dirs)]
    files = []
    for yd in year_dirs:
        files.extend(str(f) for f in yd.rglob("CVE-*.json"))
    return sorted(files)


def scan(nvd_dir: Path, dirs: list[str] | None = None, procs: int | None = None) -> pd.DataFrame:
    files = list_files(nvd_dir, dirs)
    if not files:
        raise FileNotFoundError(f"No CVE JSON files under {nvd_dir}")
    print(f"scan: parsing {len(files):,} files from {nvd_dir} ...", flush=True)
    with Pool(procs or os.cpu_count()) as pool:
        rows = [r for r in pool.imap(parse_record, files, chunksize=500) if r is not None]
    df = pd.DataFrame(rows).sort_values("cve_id", kind="stable").reset_index(drop=True)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    for c in ("cvss3_score", "cvss4_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"scan: {len(df):,} records parsed", flush=True)
    return df


def cache_path(tag: str = "snapshot") -> Path:
    return C.CACHE / f"scan_{tag}.pkl"


def load_or_scan(nvd_dir: Path, tag: str = "snapshot", dirs: list[str] | None = None,
                 force: bool = False) -> pd.DataFrame:
    p = cache_path(tag)
    if p.exists() and not force:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    df = scan(nvd_dir, dirs)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvd", type=Path, default=C.NVD_DIR)
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--tag", default="snapshot")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = load_or_scan(a.nvd, a.tag, a.dirs, a.force)
    print(d.groupby(d["published"].dt.year).size().tail(8))
