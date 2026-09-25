"""
run_all.py — reproduce every CVE number, table and figure in the paper.

    python code/cve/run_all.py                # fetch pinned snapshot if missing, then run
    python code/cve/run_all.py --nvd DIR      # use an existing clone (must be at the pinned SHA)
    python code/cve/run_all.py --rescan       # ignore the scan cache

Outputs: data/cve/*.csv, figures/cve/*.{png,pdf}, paper/generated/cve_*.tex
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze  # noqa: E402
import config as C  # noqa: E402
import figures  # noqa: E402
import scan  # noqa: E402
import texgen  # noqa: E402


def ensure_snapshot(nvd: Path):
    pinned = C.read_snapshot()["commit"]
    if not (nvd / ".git").exists():
        subprocess.run(["bash", str(Path(__file__).with_name("fetch_nvd.sh")), str(nvd)], check=True)
    head = subprocess.run(["git", "-C", str(nvd), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != pinned:
        sys.exit(f"NVD clone at {nvd} is at {head}, but the paper is pinned to {pinned}. "
                 f"Run code/cve/fetch_nvd.sh.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvd", type=Path, default=C.NVD_DIR)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--paper-dir", type=Path, default=None,
                    help="LaTeX project to copy generated/*.tex and figures into")
    a = ap.parse_args()
    t0 = time.time()
    ensure_snapshot(a.nvd)
    df = scan.load_or_scan(a.nvd, force=a.rescan)
    res = analyze.run(df)
    figures.run(res)
    texgen.run(res)
    if a.paper_dir is not None:
        (a.paper_dir / "generated").mkdir(exist_ok=True)
        (a.paper_dir / "figures").mkdir(exist_ok=True)
        for f in C.TEX_DIR.glob("cve_*.tex"):
            shutil.copy2(f, a.paper_dir / "generated" / f.name)
        for f in C.FIG_DIR.glob("cve_*.pdf"):
            shutil.copy2(f, a.paper_dir / "figures" / f.name)
        print(f"copied generated TeX + figures to {a.paper_dir}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
