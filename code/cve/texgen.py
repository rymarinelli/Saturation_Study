"""
texgen.py — LaTeX fragments generated from the analysis outputs, so every CVE
number in the paper traces to a CSV in data/cve/.

  paper/generated/cve_numbers.tex    \\newcommand macros used in the text
  paper/generated/cve_tables.tex     retained-count table + robustness table
  paper/generated/cve_lexicon.tex    appendix: keyword list, BM25 query, OWASP rules
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from lexicon import (DENY_PRODUCTS, LLM_QUERY_TERMS, OWASP_LLM, TIER_1_PATTERNS,
                     TIER_2_CONTEXT, TIER_2_FRAMEWORKS)

DEF_LABELS = {
    "intersection": r"K$\cap$B (primary)",
    "keyword": "K",
    "bm25": "B",
    "intersection_tier1": r"K$\cap$B, Tier 1",
    "keyword_tier1": "K, Tier 1",
    "bm25_0.5x": r"B, $0.5\times$",
    "bm25_2x": r"B, $2\times$",
    "intersection_0.5x": r"K$\cap$B, $0.5\times$",
    "intersection_2x": r"K$\cap$B, $2\times$",
    "intersection_tier1_0.5x": r"K$\cap$B, Tier 1, $0.5\times$",
    "intersection_tier1_2x": r"K$\cap$B, Tier 1, $2\times$",
}


def n(v) -> str:
    return f"{int(v):,}".replace(",", "{,}")


def pct(v, d=2) -> str:
    return f"{100 * v:.{d}f}"


def num(v, d=2) -> str:
    return f"{v:.{d}f}"


def pval(p: float) -> str:
    if p <= 0 or p < 1e-300:
        return r"$<10^{-300}$"
    if p < 1e-4:
        return rf"$<10^{{{int(np.floor(np.log10(p))) + 1}}}$"
    return f"{p:.3f}"


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
           "#": r"\#", "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\~{}",
           "|": r"\textbar{}\allowbreak{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def numbers(res: dict) -> str:
    info = res["info"]
    sh = res["share_by_year"]
    con = res["share_contrasts"].set_index(["definition", "window"])
    cv = res["cvss_shift"].set_index(["definition", "variant"])
    cov = res["cvss_coverage"]
    ow = res["owasp_counts"].set_index("category")
    ag = res["filter_agreement"].set_index("year")
    rob = res["robustness"]
    idy = res["id_year_contributions"]
    years = sorted(sh["year"].unique())
    y0, y1 = years[0], years[-1]

    def share(defn, y):
        return sh[(sh["definition"] == defn) & (sh["year"] == y)].iloc[0]

    m = {}
    snap = C.read_snapshot()
    m["cveSnapshotCommit"] = snap["commit"][:7]
    m["cveSnapshotDate"] = pd.Timestamp(snap["commit_date"]).strftime("%-d %B %Y")
    m["cveWindowLastDay"] = (C.window_end() - pd.Timedelta(days=1)).strftime("%-d %B %Y")
    m["cveLagCutLastDay"] = (C.LAG_CUT - pd.Timedelta(days=1)).strftime("%-d %B %Y")
    m["cveYearFirst"], m["cveYearLast"] = str(y0), str(y1)
    m["cveNPop"] = n(info["n_population"])
    m["cveNRejected"] = n(info["n_rejected"])
    m["cveNKeyword"] = n(info["n_keyword"])
    m["cveNKeywordTierOne"] = n(info["n_keyword_tier1"])
    m["cveNBM"] = n(info["n_bm25"])
    m["cveNBoth"] = n(info["n_intersection"])
    m["cveNBothTierOne"] = n(info["n_intersection_tier1"])
    m["cveBMThreshold"] = num(info["threshold_bm25"], 2)
    m["cveJaccard"] = pct(ag.loc["all", "jaccard"], 1)
    m["cveJaccardMin"] = pct(ag.loc[[y for y in years], "jaccard"].min(), 1)
    m["cveJaccardMax"] = pct(ag.loc[[y for y in years], "jaccard"].max(), 1)
    m["cveNKeywordOnly"] = n(ag.loc["all", "keyword_only"])
    m["cveNBMOnly"] = n(ag.loc["all", "bm25_only"])

    for key, defn in (("", "intersection"), ("TierOne", "intersection_tier1")):
        for tag, y in (("First", y0), ("Last", y1)):
            s = share(defn, y)
            m[f"cveShare{key}{tag}"] = pct(s["mean"], 2 if key == "" else 3)
            m[f"cveShare{key}{tag}Lo"] = pct(s["lo"], 2 if key == "" else 3)
            m[f"cveShare{key}{tag}Hi"] = pct(s["hi"], 2 if key == "" else 3)
            m[f"cveK{key}{tag}"] = n(s["k"])
            m[f"cveN{key}{tag}"] = n(s["n"])
        c = con.loc[(defn, "full")]
        m[f"cveRatio{key}"] = num(c["ratio_median"], 1)
        m[f"cveRatio{key}Lo"] = num(c["ratio_lo"], 1)
        m[f"cveRatio{key}Hi"] = num(c["ratio_hi"], 1)
        m[f"cvePIncrease{key}"] = num(c["p_increase"], 3)
    lc = con.loc[("intersection", "lagcut")]
    m["cveShareLagCutLast"] = pct(lc["share_to"])
    m["cveRatioLagCut"] = num(lc["ratio_median"], 1)
    m["cveRatioLagCutLo"] = num(lc["ratio_lo"], 1)
    m["cveRatioLagCutHi"] = num(lc["ratio_hi"], 1)
    full = con.xs("full", level="window")
    m["cvePIncreaseMin"] = num(full["p_increase"].min(), 3)
    m["cveRatioMin"] = num(full["ratio_median"].min(), 1)
    m["cveRatioMax"] = num(full["ratio_median"].max(), 1)

    p = cv.loc[("intersection", "v3")]
    m["cveMedLLM"], m["cveMedOther"] = num(p["median_llm"], 1), num(p["median_other"], 1)
    m["cveHL"], m["cveHLLo"], m["cveHLHi"] = num(p["hl_shift"], 1), num(p["hl_lo"], 1), num(p["hl_hi"], 1)
    m["cvePsup"], m["cvePsupLo"], m["cvePsupHi"] = num(p["p_superiority"]), num(p["psup_lo"]), num(p["psup_hi"])
    m["cveRankBiserial"] = num(p["rank_biserial"])
    m["cveMWp"] = pval(p["mw_p"])
    m["cveNLLMvThree"], m["cveNOthervThree"] = n(p["n_llm"]), n(p["n_other"])
    m["cveTiePairs"] = pct(p["p_tie_pairs"], 1)
    m["cveNBoot"] = n(C.N_BOOT)
    v3 = cv.xs("v3", level="variant")
    m["cveHLMin"], m["cveHLMax"] = num(v3["hl_shift"].min(), 1), num(v3["hl_shift"].max(), 1)
    m["cvePsupMin"], m["cvePsupMax"] = num(v3["p_superiority"].min()), num(v3["p_superiority"].max())
    for key, var in (("LagCut", "v3_lagcut"), ("NVD", "v3_nvd_primary_only"),
                     ("CNA", "v3_cna_secondary_only"), ("VFour", "v4")):
        r = cv.loc[("intersection", var)]
        m[f"cveHL{key}"] = num(r["hl_shift"], 1)
        m[f"cveHL{key}Lo"] = num(r["hl_lo"], 1)
        m[f"cveHL{key}Hi"] = num(r["hl_hi"], 1)
        m[f"cvePsup{key}"] = num(r["p_superiority"])
        m[f"cveN{key}"] = n(r["n_llm"])
    yrs = cv[cv.index.get_level_values("variant").str.startswith("v3_year_")]
    m["cveHLYearMin"], m["cveHLYearMax"] = num(yrs["hl_shift"].min(), 1), num(yrs["hl_shift"].max(), 1)

    ca = cov[cov["year"].astype(str) == "all"].set_index("group")
    for g, key in (("llm", "LLM"), ("other", "Other")):
        r = ca.loc[g]
        m[f"cveVThreePct{key}"] = pct(r["v3"] / r["n"], 1)
        m[f"cveVFourOnlyPct{key}"] = pct(r["v4_only"] / r["n"], 1)
        m[f"cveUnscoredPct{key}"] = pct((r["unscored"] + r["v2_only"]) / r["n"], 1)
        m[f"cveNVDPrimaryPct{key}"] = pct(r["v3_nvd_primary"] / r["v3"], 1)
        m[f"cveVThreeZeroPct{key}"] = pct(r["v3_version_3.0"] / r["v3"], 1)

    m["cveNUnmapped"] = n(ow.loc["Unmapped", "n_cves"])
    m["cveUnmappedPct"] = pct(ow.loc["Unmapped", "n_cves"] / info["n_intersection"], 0)
    small = ow[(ow.index != "Unmapped") & (ow["n_cves"] < C.MIN_OWASP_N)]
    m["cveOwaspMinN"] = str(C.MIN_OWASP_N)
    m["cveOwaspNSmall"] = str(len(small))
    m["cveOwaspSmallList"] = ", ".join(c.split("_")[0] for c in small.index)
    col = idy[str(y0)] if str(y0) in idy.columns else idy[y0]
    m["cveOldIdPctFirst"] = pct(col.drop(index=str(y0), errors="ignore").sum() / col.sum(), 0)

    # Composition of the first year of the primary set: Tier-2 (ML-framework) CVEs.
    pop, defs = res["pop"], res["defs"]
    first = (pop["year"] == y0).to_numpy()
    t2 = defs["intersection"] & ~defs["intersection_tier1"] & first
    m["cveKTierTwoFirst"] = n(t2.sum())
    m["cveTFPctTierTwoFirst"] = pct(pop.loc[t2, "text"].str.contains("tensorflow", case=False).mean(), 0)
    later = yrs[~yrs.index.get_level_values("variant").isin([f"v3_year_{y0}"])]
    m["cveHLYearFirst"] = num(cv.loc[("intersection", f"v3_year_{y0}"), "hl_shift"], 1)
    m["cveHLYearLaterMin"] = num(later["hl_shift"].min(), 1)
    m["cveHLYearLaterMax"] = num(later["hl_shift"].max(), 1)
    oth = cov[cov["group"] == "other"].set_index(cov.loc[cov["group"] == "other", "year"].astype(str))
    for tag, y in (("First", y0), ("Last", y1)):
        r = oth.loc[str(y)]
        m[f"cveNVDPrimaryPctOther{tag}"] = pct(r["v3_nvd_primary"] / r["n"], 0)

    lines = ["% Auto-generated by code/cve/texgen.py -- do not edit by hand.",
             f"% Snapshot {snap['commit']} ({snap['commit_date']})."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()]
    return "\n".join(lines) + "\n"


def tables(res: dict) -> str:
    rt = res["retained_counts"]
    rob = res["robustness"].set_index("definition")
    info = res["info"]
    years = [y for y in rt["year"] if y != "all"]
    last_partial = C.window_end() < pd.Timestamp(f"{max(years) + 1}-01-01", tz="UTC")

    out = ["% Auto-generated by code/cve/texgen.py -- do not edit by hand.",
           r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{CVEs retained per publication year (NVD snapshot \cveSnapshotCommit). "
           r"Keyword: regex filter (Tier~1+2). BM25: the " + n(info["n_keyword"]) +
           r" highest-scoring CVEs (score $\geq$ \cveBMThreshold), i.e.\ the same retention budget as the keyword filter. "
           r"Both: the analysis set. Share: posterior mean and 95\% credible interval, Jeffreys prior. "
           r"Rejected records excluded.}",
           r"\label{tab:cve-retained}",
           r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
           r"Year & All CVEs & Keyword & BM25 & Both & Jaccard & Share of all (\%) \\", r"\midrule"]
    for _, r in rt.iterrows():
        y = r["year"]
        lab = "All" if y == "all" else (f"{y}$^\\dagger$" if (last_partial and y == max(years)) else str(y))
        if y == "all":
            out.append(r"\midrule")
        out.append(f"{lab} & {n(r['total'])} & {n(r['keyword'])} & {n(r['bm25'])} & {n(r['both'])} & "
                   f"{r['jaccard']:.2f} & {100 * r['share_mean']:.2f} [{100 * r['share_lo']:.2f}, {100 * r['share_hi']:.2f}] \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if last_partial:
        out.append(r"\\[2pt]{\scriptsize $^\dagger$1 January to \cveWindowLastDay.}")
    out += [r"\end{table}", ""]

    out += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
            r"\caption{Headline CVE results under alternative relevance definitions. K: keyword filter; B: BM25 "
            r"at the keyword filter's retention budget, or $0.5\times$/$2\times$ it; Tier~1: LLM-specific "
            r"keyword matches only. Share: posterior mean "
            r"in the first and last year of the window; $P(\uparrow)$: posterior probability that the share "
            r"increased (" + str(max(years)) + r": to \cveWindowLastDay). HL: Hodges--Lehmann shift of CVSS v3.x base scores, LLM-related minus all other CVEs, "
            r"with 95\% bootstrap interval (" + n(C.N_BOOT) + r" resamples); $\hat{P}$: probability that a random "
            r"LLM-related CVE scores higher than a random other CVE (ties count one half).}",
            r"\label{tab:cve-robustness}",
            r"\begin{tabular}{@{}lrrrrl r@{}}", r"\toprule",
            r"Definition & $n$ & \multicolumn{2}{c}{Share (\%)} & $P(\uparrow)$ & HL [95\% CI] & $\hat{P}$ \\",
            r"\cmidrule(lr){3-4}",
            rf" & & {min(years)} & {max(years)} & & & \\", r"\midrule"]
    for d in DEF_LABELS:
        if d not in rob.index:
            continue
        r = rob.loc[d]
        out.append(f"{DEF_LABELS[d]} & {n(info['n_' + d])} & {100 * r['share_from']:.2f} & {100 * r['share_to']:.2f} & "
                   f"{r['p_increase']:.3f} & {r['hl_shift']:.1f} [{r['hl_lo']:.1f}, {r['hl_hi']:.1f}] & {r['p_superiority']:.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def lexicon_appendix() -> str:
    def lst(items):
        return r",\allowbreak{} ".join(r"\texttt{" + tex_escape(s) + "}" for s in items)

    out = ["% Auto-generated by code/cve/texgen.py -- do not edit by hand.",
           r"\subsection{CVE relevance filters and OWASP mapping}\label{app:cve-lexicon}",
           r"All patterns are case-insensitive Python regular expressions applied to the English "
           r"description concatenated with the record's CPE match strings. A record is \emph{Tier~1} if "
           r"it matches any Tier-1 pattern, \emph{Tier~2} if it matches a Tier-2 framework pattern "
           r"\emph{and} the ML-context pattern, and it is discarded if it matches the deny-list. "
           r"All rules were written by hand; no language model was used to label or map CVEs.",
           "", r"{\scriptsize\raggedright",
           r"\textbf{Tier 1 (" + str(len(TIER_1_PATTERNS)) + r" patterns).} " + lst(TIER_1_PATTERNS) + r"\par",
           r"\textbf{Tier 2 frameworks.} " + lst(TIER_2_FRAMEWORKS) +
           r"; \textbf{ML context:} " + lst([TIER_2_CONTEXT]) + r"\par",
           r"\textbf{Deny-list.} " + lst(DENY_PRODUCTS) + r"\par",
           r"\textbf{BM25 query (" + str(len(LLM_QUERY_TERMS)) + r" tokens; Okapi, $k_1=" + f"{C.BM25_K1}" +
           r"$, $b=" + f"{C.BM25_B}" + r"$).} " + lst(LLM_QUERY_TERMS) + r"\par",
           r"\textbf{OWASP Top~10 for LLM Applications (2025).} A CVE is assigned every category whose "
           r"description pattern matches or whose CWE set intersects the record's CWEs; Tier-2 records "
           r"with no match default to LLM03.\par"]
    for cat, (pat, cwes) in OWASP_LLM.items():
        cw = (" CWEs: " + ", ".join(sorted(cwes, key=lambda c: int(c.split("-")[1])))) if cwes else ""
        out.append(r"\textbf{" + tex_escape(cat.replace("_", " ", 1).split("_")[0]) + r"}: \texttt{" +
                   tex_escape(pat) + "}" + tex_escape(cw) + r"\par")
    out += ["}", ""]
    return "\n".join(out)


def run(res: dict):
    C.TEX_DIR.mkdir(parents=True, exist_ok=True)
    (C.TEX_DIR / "cve_numbers.tex").write_text(numbers(res))
    (C.TEX_DIR / "cve_tables.tex").write_text(tables(res))
    (C.TEX_DIR / "cve_lexicon.tex").write_text(lexicon_appendix())
    print(f"texgen: wrote cve_numbers/cve_tables/cve_lexicon.tex to {C.TEX_DIR}", flush=True)
