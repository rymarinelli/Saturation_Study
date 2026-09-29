"""Paper figure "Saturation Reported from Model Cards": Cybench as reported (Anthropic; pass@30 for
Claude 3.5/3.7 Sonnet, average pass@1 from Claude Sonnet 4.5 on) and
agentic-offense scores (CyberGym, Anthropic; Cyber Range, OpenAI) by model-card date.

Usage: python code/model_cards/fig_model_card_saturation.py [data/model_card_cyber_evals.csv] [figures/]

Input is the hand-collected model-card table; each row cites its card_url. Each line shows the
frontier: the highest-scoring model per card date and vendor (a smaller model released in the
same month is omitted by rule, not by hand).
"""
import re, sys, os
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False})

VENDOR_COLORS = {"OpenAI": "#10a37f", "Anthropic": "#c96a3e"}
PLOT_VENDORS = set(VENDOR_COLORS)   # the paper plots these two; other labs' rows stay in the table only
LABEL_NUDGE = {}   # {"model name": (days_right, score_up)} to move one label by hand

def _num(s):
    try: return float(str(s).strip())
    except (ValueError, TypeError): return float("nan")

def parse_cybench(s):
    """'15/34 (44%)' -> 44.0;  '60' -> 60.0;  '' -> NaN."""
    s = str(s).strip()
    m = re.search(r"\(([0-9.]+)%\)", s)
    if m: return float(m.group(1))
    m = re.search(r"^([0-9.]+)", s)
    return float(m.group(1)) if m else float("nan")

METRICS = [  # (column, parser, series label)
    ("cybench_score", parse_cybench, "Cybench"),
    ("cybergym_score_pct", _num, "CyberGym"),
    ("openai_cyber_range_pass_rate_pct", _num, "Cyber Range"),
]

def load(path):
    mc = pd.read_csv(path, dtype=str).fillna("")
    rows = []
    for _, r in mc.iterrows():
        if not r["model_name"].strip() or not r["card_date"].strip(): continue
        if r["lab"].strip() not in PLOT_VENDORS: continue
        for col, parse, series in METRICS:
            v = parse(r[col])
            if pd.notna(v):
                rows.append(dict(model=r["model_name"].strip(), vendor=r["lab"].strip(),
                                 date=pd.to_datetime(r["card_date"].strip()), series=series, value=v))
    d = pd.DataFrame(rows)
    return d.loc[d.groupby(["vendor", "series", "date"])["value"].idxmax()].sort_values("date")

def _place(ax, it, tx, ty, ha, va, fs, leader):
    ddays, dval = LABEL_NUDGE.get(it["text"], (0, 0))
    ax.annotate(it["text"], xy=(it["x"], it["y"]), xytext=(tx + pd.Timedelta(days=ddays), ty + dval),
                textcoords="data", ha=ha, va=va, fontsize=fs, color=it["color"],
                arrowprops=(dict(arrowstyle="-", color=it["color"], lw=0.6, alpha=0.5) if leader else None))

def _label_points(ax, items, fs=7.5, drop=0.09, right_gap_days=16, right_step=0.10,
                  fan_days=26, fan_drop=0.12, fan_step=0.10):
    """Isolated points: label below. Final same-date cluster: stacked right with leaders.
    Interior same-date clusters: fanned below with leaders."""
    if not items: return
    items = sorted(items, key=lambda d: d["x"])
    clusters = [[items[0]]]
    for it in items[1:]:
        if (it["x"] - clusters[-1][-1]["x"]).days <= 25: clusters[-1].append(it)
        else: clusters.append([it])
    ylo, yhi = ax.get_ylim(); span = yhi - ylo
    last_x = max(it["x"] for it in items)
    for cl in clusters:
        if len(cl) == 1:
            it = cl[0]; _place(ax, it, it["x"], it["y"] - span * drop, "center", "top", fs, leader=False)
        elif any(it["x"] == last_x for it in cl):
            cl = sorted(cl, key=lambda d: -d["y"])
            start = min(max(d["y"] for d in cl), yhi - span * 0.14)
            lx = cl[0]["x"] + pd.Timedelta(days=right_gap_days)
            for k, it in enumerate(cl):
                _place(ax, it, lx, start - k * span * right_step, "left", "center", fs, leader=True)
        else:
            cl = sorted(cl, key=lambda d: -d["y"]); n = len(cl); base = min(d["y"] for d in cl)
            for k, it in enumerate(cl):
                lx = it["x"] + pd.Timedelta(days=(k - (n - 1) / 2) * fan_days)
                _place(ax, it, lx, base - span * (fan_drop + fan_step * k), "center", "top", fs, leader=True)

def _pad_xlim(ax, left=0.04, right=0.20):
    lo, hi = ax.get_xlim(); span = hi - lo
    ax.set_xlim(lo - span * left, hi + span * right)

def _panel(ax, d, series, title, ylabel, fs, legend_fs):
    items = []
    for (vendor, s), g in d[d["series"].isin(series)].groupby(["vendor", "series"]):
        color = VENDOR_COLORS.get(vendor, "#888")
        ls = "--" if s == "Cyber Range" else "-"
        label = vendor if len(series) == 1 else f"{vendor} — {s}"
        ax.plot(g["date"], g["value"], "o", color=color, zorder=5)
        ax.plot(g["date"], g["value"], color=color, lw=2.2 if len(series) == 1 else 2, ls=ls, label=label, zorder=5)
        items += [{"x": r["date"], "y": r["value"], "text": r["model"], "color": color} for _, r in g.iterrows()]
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(100, color="#bbb", lw=0.8, ls="--", zorder=1)
    _label_points(ax, items, fs=fs); _pad_xlim(ax)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=legend_fs, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

def main(src, figdir):
    d = load(src)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.5})
    _panel(ax1, d, ["Cybench"], "Cybench over time, as reported in each card  (100 % line = benchmark saturation)",
           "Cybench score (%)", 7.5, 9)
    _panel(ax2, d, ["CyberGym", "Cyber Range"], "Agentic offense: CyberGym (Anthropic) and Cyber Range (OpenAI)",
           "Score (%)", 7, 8.5)
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(f"{figdir}/model_card_saturation.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{figdir}/model_card_saturation.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    print(f"wrote {figdir}/model_card_saturation.(png|pdf) from {len(d)} frontier points")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/model_card_cyber_evals.csv",
         sys.argv[2] if len(sys.argv) > 2 else "figures")
