"""
figures.py — the two CVE figures in the paper.

  cve_share_quarterly  quarterly share of LLM-related CVEs, posterior mean and
                       95% credible band, primary definition vs Tier-1 only.
  cve_owasp_cvss       CVSS v3 base-score distribution per OWASP LLM category
                       (primary set), n per category, small categories merged.

Descriptive only: no model-release overlays.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lexicon import OWASP_ORDER  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 300, "font.family": "DejaVu Sans",
})

OWASP_SHORT = {
    "LLM01_Prompt_Injection": "LLM01 Prompt injection",
    "LLM02_Sensitive_Info_Disclosure": "LLM02 Sensitive info",
    "LLM03_Supply_Chain": "LLM03 Supply chain",
    "LLM04_Data_Model_Poisoning": "LLM04 Data/model poisoning",
    "LLM05_Improper_Output_Handling": "LLM05 Improper output",
    "LLM06_Excessive_Agency": "LLM06 Excessive agency",
    "LLM07_System_Prompt_Leakage": "LLM07 System prompt leak",
    "LLM08_Vector_Embedding_Weakness": "LLM08 Vector/embedding",
    "LLM09_Misinformation": "LLM09 Misinformation",
    "LLM10_Unbounded_Consumption": "LLM10 Unbounded consumption",
}

LABELS = {
    "intersection": "Keyword ∩ BM25 (Tier 1+2, primary)",
    "intersection_tier1": "Keyword ∩ BM25, Tier 1 only",
}


def _save(fig, name, tight=True):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}", bbox_inches="tight" if tight else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def share_figure(quarterly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for name, col in (("intersection", BLUE), ("intersection_tier1", ORANGE)):
        d = quarterly[quarterly["definition"] == name].reset_index(drop=True)
        x = np.arange(len(d))
        ax.fill_between(x, d["lo"] * 100, d["hi"] * 100, color=col, alpha=0.18, lw=0)
        full = ~d["partial"]
        ax.plot(x[full], d.loc[full, "mean"] * 100, color=col, lw=1.6, label=LABELS[name])
        if (~full).any():
            ax.plot(x[~full], d.loc[~full, "mean"] * 100, "o", ms=3.5, mfc="white", mec=col, mew=1.2)
    qs = quarterly.loc[quarterly["definition"] == "intersection", "quarter"].tolist()
    ticks = [i for i, q in enumerate(qs) if q.endswith("Q1")]
    ax.set_xticks(ticks, [qs[i][:4] for i in ticks])
    ax.set_ylabel("Share of all published CVEs (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.4)
    _save(fig, "cve_share_quarterly")


def owasp_figure(owasp_exp: pd.DataFrame, owasp_counts: pd.DataFrame):
    e = owasp_exp.dropna(subset=["cvss3_score"])
    other = [g for g in e["plot_group"].unique() if str(g).startswith("Other")]
    order = [c for c in OWASP_ORDER if c in set(e["plot_group"])] + other + \
            (["Unmapped"] if "Unmapped" in set(e["plot_group"]) else [])
    data = [e.loc[e["plot_group"] == g, "cvss3_score"].to_numpy() for g in order]
    n_cves = {g: int(owasp_exp.loc[owasp_exp["plot_group"] == g, "cve_id"].nunique()) for g in order}

    merged = owasp_counts.loc[owasp_counts["plot_group"].str.startswith("Other"), "category"]
    other_label = "Other (" + "/".join(c.split("_")[0][3:] if i else c.split("_")[0] for i, c in enumerate(merged)) + ")"

    def lab(g):
        if g.startswith("Other"):
            base = other_label
        else:
            base = OWASP_SHORT.get(g, g)
        return f"{base} ({n_cves[g]})"

    # Fixed column-width canvas (no tight bbox) so 7pt labels print at 7pt.
    fig, ax = plt.subplots(figsize=(3.5, 0.22 * len(order) + 0.55))
    fig.subplots_adjust(left=0.53, right=0.97, top=0.98, bottom=0.42 / (0.22 * len(order) + 0.55))
    bp = ax.boxplot(data[::-1], orientation="horizontal", widths=0.55, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.2},
                    boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE, "lw": 0.8},
                    whiskerprops={"color": BLUE, "lw": 0.8}, capprops={"color": BLUE, "lw": 0.8},
                    flierprops={"marker": "o", "ms": 2, "mfc": BLUE, "mec": "none", "alpha": 0.5})
    ax.set_yticks(range(1, len(order) + 1), [lab(g) for g in order[::-1]], fontsize=6.5)
    ax.set_xlim(0, 10.2)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("CVSS v3.x base score")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "cve_owasp_cvss", tight=False)
    return bp


def run(res: dict):
    share_figure(res["share_by_quarter"])
    owasp_figure(res["owasp_exp"], res["owasp_counts"])
    print(f"figures: wrote to {C.FIG_DIR}", flush=True)
