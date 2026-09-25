"""
analyze.py — every CVE number in the paper, written to data/cve/*.csv and
paper/generated/cve_*.tex.

Population: NVD records published in [WINDOW_START, window_end()), excluding
vulnStatus == "Rejected" and records without an English description.

Relevance definitions (all automated, no labels):
  keyword            regex filter, Tier 1 + Tier 2          (lexicon.keyword_relevance)
  bm25               top-|keyword| CVEs by BM25 score        ("equal budget")
  intersection       keyword AND bm25  -> PRIMARY analysis set
  *_tier1            the same, restricted to Tier-1 (LLM-specific) keyword matches
  *_0.5x / *_2x      BM25 budget at 0.5x / 2x the keyword count
"""

from __future__ import annotations

import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as C
import stats as S
from bm25 import budget_threshold, fit_bm25, score_corpus
from lexicon import OWASP_ORDER, categorize_owasp, tokenize

PRIMARY = "intersection"


# ─────────────────────────────────────────────────────────────────────────────
# Population + BM25
# ─────────────────────────────────────────────────────────────────────────────
def population(scan_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    end = C.window_end()
    in_win = scan_df["published"].ge(C.WINDOW_START) & scan_df["published"].lt(end)
    w = scan_df[in_win]
    rejected = w["vuln_status"].eq("Rejected")
    no_desc = ~w["has_en_desc"] & ~rejected
    pop = w[~rejected & w["has_en_desc"]].copy().reset_index(drop=True)
    pop["year"] = pop["published"].dt.year
    pop["quarter"] = pop["published"].dt.tz_localize(None).dt.to_period("Q")
    info = {"window_start": C.WINDOW_START.date().isoformat(),
            "window_end_exclusive": end.date().isoformat(),
            "n_in_window_records": int(in_win.sum()),
            "n_rejected": int(rejected.sum()), "n_no_en_desc": int(no_desc.sum()),
            "n_population": len(pop)}
    return pop, info


def bm25_scores(pop: pd.DataFrame) -> np.ndarray:
    cache = C.CACHE / "bm25_scores.pkl"
    key = (C.read_snapshot()["commit"], C.window_end().isoformat(), len(pop),
           C.BM25_VARIANT, C.BM25_K1, C.BM25_B)
    if cache.exists():
        with open(cache, "rb") as fh:
            k, ids, sc = pickle.load(fh)
        if k == key and np.array_equal(ids, pop["cve_id"].to_numpy()):
            return sc
    print(f"bm25: tokenizing {len(pop):,} documents ...", flush=True)
    with Pool(os.cpu_count()) as pool:
        toks = pool.map(tokenize, pop["text"].tolist(), chunksize=2000)
    print("bm25: fitting + scoring ...", flush=True)
    model = fit_bm25(toks, variant=C.BM25_VARIANT, k1=C.BM25_K1, b=C.BM25_B)
    sc = score_corpus(model)
    with open(cache, "wb") as fh:
        pickle.dump((key, pop["cve_id"].to_numpy(), sc), fh)
    return sc


def definitions(pop: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    K = pop["keyword"].ne("").to_numpy()
    T1 = pop["keyword"].eq("adversarial").to_numpy()
    sc = pop["bm25_score"].to_numpy()
    defs, thr = {"keyword": K, "keyword_tier1": T1}, {}
    for m in C.BUDGET_MULTS:
        t = budget_threshold(sc, int(round(m * K.sum())))
        B = sc >= t
        sfx = "" if m == 1.0 else f"_{m:g}x"
        thr[f"bm25{sfx}"] = t
        defs[f"bm25{sfx}"] = B
        defs[f"intersection{sfx}"] = K & B
        defs[f"intersection_tier1{sfx}"] = K & B & T1
    return defs, thr


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
def share_tables(pop, defs, years):
    rows, contrasts = [], []
    tot = pop.groupby("year").size()
    lag = pop["published"].lt(C.LAG_CUT).to_numpy()
    for name, m in defs.items():
        k = pd.Series(m).groupby(pop["year"].to_numpy()).sum()
        for y in years:
            rows.append({"definition": name, "year": y, **S.beta_posterior(int(k.get(y, 0)), int(tot[y]))})
        for win, sel in (("full", np.ones(len(pop), bool)), ("lagcut", lag)):
            y0, y1 = years[0], years[-1]
            s0 = sel & (pop["year"].to_numpy() == y0)
            s1 = sel & (pop["year"].to_numpy() == y1)
            k0, n0, k1, n1 = int(m[s0].sum()), int(s0.sum()), int(m[s1].sum()), int(s1.sum())
            contrasts.append({"definition": name, "window": win, "year_from": y0, "year_to": y1,
                              "k_from": k0, "n_from": n0, "k_to": k1, "n_to": n1,
                              "share_from": (k0 + .5) / (n0 + 1), "share_to": (k1 + .5) / (n1 + 1),
                              **S.share_contrast(k0, n0, k1, n1, f"contrast/{name}/{win}")})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def quarterly_shares(pop, defs, names=("intersection", "intersection_tier1", "keyword", "bm25")):
    tot = pop.groupby("quarter").size()
    rows = []
    for name in names:
        k = pd.Series(defs[name]).groupby(pop["quarter"].to_numpy()).sum()
        for q, n in tot.items():
            rows.append({"definition": name, "quarter": str(q),
                         "partial": q.end_time.tz_localize("UTC") >= C.window_end(),
                         **S.beta_posterior(int(k.get(q, 0)), int(n))})
    return pd.DataFrame(rows)


def agreement_table(pop, defs, years):
    K, B = defs["keyword"], defs["bm25"]
    yr = pop["year"].to_numpy()
    rows = []
    for y in list(years) + ["all"]:
        s = np.ones(len(pop), bool) if y == "all" else yr == y
        k, b = K & s, B & s
        inter, union = int((k & b).sum()), int((k | b).sum())
        rows.append({"year": y, "total": int(s.sum()), "keyword": int(k.sum()), "bm25": int(b.sum()),
                     "both": inter, "keyword_only": int((k & ~b).sum()), "bm25_only": int((b & ~k).sum()),
                     "union": union, "jaccard": inter / union if union else np.nan})
    for m in C.BUDGET_MULTS:
        if m == 1.0:
            continue
        b = defs[f"bm25_{m:g}x"]
        inter, union = int((K & b).sum()), int((K | b).sum())
        rows.append({"year": f"all_bm25_{m:g}x", "total": len(pop), "keyword": int(K.sum()),
                     "bm25": int(b.sum()), "both": inter, "keyword_only": int((K & ~b).sum()),
                     "bm25_only": int((b & ~K).sum()), "union": union, "jaccard": inter / union})
    return pd.DataFrame(rows)


def retained_table(pop, defs, years, shares):
    ag = agreement_table(pop, defs, years)
    ag = ag[ag["year"].isin(list(years) + ["all"])].copy()
    ps = shares[shares["definition"] == PRIMARY].set_index("year")
    n_all = len(pop)
    k_all = int(defs[PRIMARY].sum())
    overall = S.beta_posterior(k_all, n_all)
    ag["share_mean"] = [ps.loc[y, "mean"] if y != "all" else overall["mean"] for y in ag["year"]]
    ag["share_lo"] = [ps.loc[y, "lo"] if y != "all" else overall["lo"] for y in ag["year"]]
    ag["share_hi"] = [ps.loc[y, "hi"] if y != "all" else overall["hi"] for y in ag["year"]]
    return ag


def cvss_tables(pop, defs, years):
    v3 = pop["cvss3_score"].to_numpy()
    v4 = pop["cvss4_score"].to_numpy()
    src = pop["cvss3_source"].to_numpy()
    lag = pop["published"].lt(C.LAG_CUT).to_numpy()
    yr = pop["year"].to_numpy()
    has3, has4 = ~np.isnan(v3), ~np.isnan(v4)
    rows = []

    def add(name, variant, m, sel, score):
        x, y = score[m & sel], score[~m & sel]
        rows.append({"definition": name, "variant": variant,
                     **S.cvss_shift(x, y, f"cvss/{name}/{variant}")})

    for name, m in defs.items():
        add(name, "v3", m, has3, v3)
    add(PRIMARY, "v3_lagcut", defs[PRIMARY], has3 & lag, v3)
    add(PRIMARY, "v3_nvd_primary_only", defs[PRIMARY], has3 & (src == "Primary"), v3)
    add(PRIMARY, "v3_cna_secondary_only", defs[PRIMARY], has3 & (src == "Secondary"), v3)
    add(PRIMARY, "v4", defs[PRIMARY], has4, v4)
    for y in years:
        add(PRIMARY, f"v3_year_{y}", defs[PRIMARY], has3 & (yr == y), v3)
    return pd.DataFrame(rows)


def cvss_coverage(pop, defs, years):
    m = defs[PRIMARY]
    has3 = pop["cvss3_score"].notna().to_numpy()
    has4 = pop["cvss4_score"].notna().to_numpy()
    prim = pop["cvss3_source"].eq("Primary").to_numpy()
    v30 = pop["cvss3_version"].eq("3.0").to_numpy()
    v2 = pop["has_cvss2"].to_numpy()
    rows = []
    for grp, g in (("llm", m), ("other", ~m)):
        for y in list(years) + ["all"]:
            s = g & (np.ones(len(pop), bool) if y == "all" else pop["year"].to_numpy() == y)
            n = int(s.sum())
            rows.append({"group": grp, "year": y, "n": n,
                         "v3": int((s & has3).sum()), "v3_version_3.0": int((s & v30).sum()),
                         "v3_nvd_primary": int((s & has3 & prim).sum()),
                         "v3_cna_secondary": int((s & has3 & ~prim).sum()),
                         "v4_only": int((s & ~has3 & has4).sum()),
                         "v2_only": int((s & ~has3 & ~has4 & v2).sum()),
                         "unscored": int((s & ~has3 & ~has4 & ~v2).sum())})
    df = pd.DataFrame(rows)
    df["v3_share"] = df["v3"] / df["n"]
    return df


def owasp_tables(pop, defs, years):
    m = defs[PRIMARY]
    sub = pop[m].copy()
    sub["owasp"] = [categorize_owasp(t, c.split("|") if c else [], r)
                    for t, c, r in zip(sub["text"], sub["cwes"], sub["keyword"])]
    counts = {c: int(sum(c in cats for cats in sub["owasp"])) for c in OWASP_ORDER}
    small = [c for c, n in counts.items() if n < C.MIN_OWASP_N]
    rows = []
    exp = sub[["cve_id", "year", "cvss3_score", "owasp"]].explode("owasp")
    exp["owasp"] = exp["owasp"].fillna("Unmapped")
    exp["plot_group"] = exp["owasp"].where(~exp["owasp"].isin(small), f"Other (each n<{C.MIN_OWASP_N})")
    for c in OWASP_ORDER + ["Unmapped"]:
        e = exp[exp["owasp"] == c]
        s = e["cvss3_score"].dropna()
        rows.append({"category": c, "n_cves": len(e), "n_scored_v3": len(s),
                     "median_v3": s.median() if len(s) else np.nan,
                     "q1_v3": s.quantile(.25) if len(s) else np.nan,
                     "q3_v3": s.quantile(.75) if len(s) else np.nan,
                     "plot_group": e["plot_group"].iloc[0] if len(e) else (f"Other (each n<{C.MIN_OWASP_N})" if c in small else c)})
    by_year = (exp.groupby(["owasp", "year"]).size().unstack(fill_value=0)
               .reindex(index=OWASP_ORDER + ["Unmapped"], fill_value=0).reindex(columns=years, fill_value=0))
    return pd.DataFrame(rows), by_year.reset_index().rename(columns={"index": "owasp"}), exp, sub


def id_year_table(pop, years):
    idy = pop["id_year"].where(pop["id_year"] >= 2018, 0).astype(int)
    t = pd.crosstab(idy.map(lambda v: "<2018" if v == 0 else str(v)), pop["year"])
    return t.reindex(columns=years, fill_value=0)


def analysis_set_export(pop, defs, owasp_sub):
    keep = defs["keyword"] | defs["bm25"]
    out = pop.loc[keep, ["cve_id", "published", "id_year", "keyword", "bm25_score",
                         "cvss3_score", "cvss3_version", "cvss3_source", "cvss4_score", "cwes"]].copy()
    out["published"] = out["published"].dt.strftime("%Y-%m-%d")
    out["in_keyword"] = defs["keyword"][keep]
    out["in_bm25"] = defs["bm25"][keep]
    out["in_intersection"] = defs[PRIMARY][keep]
    out["owasp"] = out["cve_id"].map(owasp_sub.set_index("cve_id")["owasp"].map("|".join)).fillna("")
    out["text_snippet"] = pop.loc[keep, "text"].str.slice(0, 200).str.replace(r"\s+", " ", regex=True)
    return out.rename(columns={"keyword": "keyword_tier"})


def robustness_table(contrasts, cvss):
    c = contrasts[contrasts["window"] == "full"].set_index("definition")
    v = cvss[cvss["variant"] == "v3"].set_index("definition")
    rows = []
    for d in c.index:
        rows.append({"definition": d, "share_from": c.loc[d, "share_from"], "share_to": c.loc[d, "share_to"],
                     "diff_lo": c.loc[d, "diff_lo"], "diff_hi": c.loc[d, "diff_hi"],
                     "p_increase": c.loc[d, "p_increase"],
                     "hl_shift": v.loc[d, "hl_shift"], "hl_lo": v.loc[d, "hl_lo"], "hl_hi": v.loc[d, "hl_hi"],
                     "p_superiority": v.loc[d, "p_superiority"], "n_llm_v3": v.loc[d, "n_llm"]})
    return pd.DataFrame(rows)


def run(scan_df: pd.DataFrame) -> dict:
    C.DATA.mkdir(parents=True, exist_ok=True)
    pop, info = population(scan_df)
    pop["bm25_score"] = bm25_scores(pop)
    years = sorted(pop["year"].unique().tolist())
    defs, thr = definitions(pop)
    info.update({f"threshold_{k}": v for k, v in thr.items()})
    info.update({f"n_{k}": int(v.sum()) for k, v in defs.items()})
    info["snapshot_commit"] = C.read_snapshot()["commit"]
    info["snapshot_commit_date"] = C.read_snapshot()["commit_date"]

    shares, contrasts = share_tables(pop, defs, years)
    quarterly = quarterly_shares(pop, defs)
    agreement = agreement_table(pop, defs, years)
    retained = retained_table(pop, defs, years, shares)
    cvss = cvss_tables(pop, defs, years)
    coverage = cvss_coverage(pop, defs, years)
    owasp, owasp_year, owasp_exp, owasp_sub = owasp_tables(pop, defs, years)
    idyear = id_year_table(pop, years)
    robust = robustness_table(contrasts, cvss)
    aset = analysis_set_export(pop, defs, owasp_sub)

    out = {"population_info": pd.DataFrame([info]).T.rename(columns={0: "value"}),
           "retained_counts": retained, "filter_agreement": agreement,
           "share_by_year": shares, "share_contrasts": contrasts, "share_by_quarter": quarterly,
           "cvss_shift": cvss, "cvss_coverage": coverage, "owasp_counts": owasp,
           "owasp_by_year": owasp_year, "id_year_contributions": idyear,
           "robustness": robust, "analysis_set": aset}
    for name, df in out.items():
        idx = name in ("population_info", "id_year_contributions")
        df.to_csv(C.DATA / f"{name}.csv", index=idx, float_format="%.6g")
    print(f"analyze: wrote {len(out)} CSVs to {C.DATA}", flush=True)
    return {"pop": pop, "defs": defs, "info": info, "owasp_exp": owasp_exp, **out}
