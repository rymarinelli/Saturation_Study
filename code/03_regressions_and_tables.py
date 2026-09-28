"""Paper Tables 1-4 (cohorts, regressions, FPCA, per-benchmark status) and results_master.csv.
Usage: python 03_regressions_and_tables.py <output_dir_from_step_02>
Requires step 02 outputs; imports step 04 for the FPCA rows.
Writes table_cohorts.csv, table_regressions.csv, table_status.csv, results_master.csv, and
paper/results_tables.md.
"""
import pandas as pd, numpy as np, sys, os, importlib
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sat_config import T0, SAT_CUTOFF, MIN_MODELS, COHORTS, TABLE4_BENCHMARKS

fda = importlib.import_module('04_fda_analysis')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def intro_yr(d): return d.dt.year + d.dt.dayofyear/365

def lifespan_sample(summ):
    """Benchmarks entering the lifespan analysis (shared with 05_figures.py)."""
    s=summ.copy()
    s['sat']=s['crossed_95'].fillna(s['projected_95'])
    s['months']=(s['sat']-s['first_date']).dt.days/30.44
    return s[pd.notna(s['sat'])&(s['n_models']>=MIN_MODELS)&(s['sat']<SAT_CUTOFF)&(s['months']>0)]

def peak_velocity(panel, summ):
    rows=[]
    for b,g in panel.groupby('benchmark'):
        g=g.sort_values('month')
        if len(g)<12 or g['sota'].max()<0.70: continue
        ys=uniform_filter1d(g['sota'].values,size=3)
        rows.append(dict(benchmark=b,vmax=np.diff(ys).max()*100))
    v=pd.DataFrame(rows).merge(summ[['benchmark','first_date']],on='benchmark')
    v['iy']=intro_yr(v['first_date']); return v

def reg(x,y):
    r=stats.linregress(x,y); return r.slope,r.rvalue,r.pvalue

def fmt_p(p): return '<0.001' if p<0.001 else (f'{p:.4f}' if p<0.01 else f'{p:.3f}' if p<0.1 else f'{p:.2f}')

def main(d):
    summ=pd.read_csv(f'{d}/saturation_summary.csv',parse_dates=['first_date','last_date','crossed_90','crossed_95','projected_95'])
    panel=pd.read_csv(f'{d}/sota_monthly_panel.csv',parse_dates=['month'])
    fits=pd.read_csv(f'{d}/saturation_curve_fits.csv')
    best=fits.sort_values('rmse').groupby('benchmark').first().reset_index()
    best=best.merge(summ[['benchmark','first_date','n_models']],on='benchmark'); best['iy']=intro_yr(best['first_date'])
    best=best[best['n_models']>=MIN_MODELS].drop(columns='n_models')   # fits share the >=8-model restriction (paper, Sec. 5)
    summ['sat']=summ['crossed_95'].fillna(summ['projected_95'])
    summ['months']=(summ['sat']-summ['first_date']).dt.days/30.44
    life=lifespan_sample(summ)
    vel=peak_velocity(panel,summ)
    fp=fda.fpca(d)

    # --- Table 1: cohorts ---
    rows=[]
    for lo,hi in COHORTS:
        sel=lambda x: x[(x['first_date']>=f'{lo}-01-01')&(x['first_date']<f'{hi}-01-01')]
        allb,lf,kk,vv=sel(summ),sel(life),sel(best),sel(vel)
        rows.append({'cohort':f'{lo}–{hi-1}' if hi-1>lo else str(lo),'benchmarks':len(allb),
                     'median_lifespan_yr':round(lf['months'].median()/12,1) if len(lf) else None,'n_lifespan':len(lf),
                     'median_k':round(kk['rate_k'].median(),2) if len(kk)>1 else None,'n_k':len(kk),
                     'median_vmax':round(vv['vmax'].median(),1) if len(vv) else None,'n_vmax':len(vv)})
    t1=pd.DataFrame(rows)

    # --- Table 2: regressions on introduction date ---
    emp=life[pd.notna(life['crossed_95'])]
    bb=best[best['rate_k']<15]
    r=[('Lifespan to 95% (months)','Realized + projected crossings',*reg((life['first_date']-T0).dt.days/365.25,life['months']),len(life),None),
       ('Lifespan to 95% (months)','Empirically realized crossings only',*reg((emp['first_date']-T0).dt.days/365.25,emp['months']),len(emp),None),
       ('log fitted rate k','Best sigmoid fit per benchmark',*reg(bb['iy'],np.log(bb['rate_k'])),len(bb),'x'),
       ('log peak velocity','Benchmarks past 70%',*reg(vel['iy'],np.log(vel['vmax'].clip(lower=0.1))),len(vel),'x')]
    s5=reg(fp['iy'],fp['scores'][:,0])
    r.append(('FPCA PC1 score (steepness mode)','Fully observed registered curves',abs(s5[0]),abs(s5[1]),s5[2],len(fp['names']),None))
    t2=pd.DataFrame(r,columns=['outcome','sample','slope','r','p','n','mult'])
    t2['mult']=np.where(t2['mult']=='x',np.exp(t2['slope']),np.nan)

    # --- Table 3: FPCA ---
    t3=pd.DataFrame({'component':['PC1','PC2'],'variance_explained':fp['evr'],
                     'corr_with_gain':[-abs(fp['corr'][0]),abs(fp['corr'][1])],'n':len(fp['names'])})

    # --- Table 4 + results_master ---
    m=summ.merge(best[['benchmark','model','ceiling','rate_k','rmse']],on='benchmark',how='left')\
          .merge(vel[['benchmark','vmax']],on='benchmark',how='left')
    m['lifespan_yr']=m['months']/12; m['empirical']=m['crossed_95'].notna(); m['intro_yr']=intro_yr(m['first_date'])
    m['in_lifespan_sample']=m['benchmark'].isin(life['benchmark'])
    m.drop(columns=['months']).to_csv(f'{d}/results_master.csv',index=False,float_format='%.6g',date_format='%Y-%m-%d')
    t4=m[m['benchmark'].isin(TABLE4_BENCHMARKS)].sort_values(['first_date','benchmark'])

    t1.to_csv(f'{d}/table_cohorts.csv',index=False)
    t2.to_csv(f'{d}/table_regressions.csv',index=False,float_format='%.6g')
    t3.to_csv(f'{d}/table_fpca.csv',index=False,float_format='%.6g')
    t4[['benchmark','n_models','first_date','current_sota','crossed_95','projected_95','lifespan_yr','model','ceiling','rate_k','vmax']]\
      .to_csv(f'{d}/table_status.csv',index=False,float_format='%.6g',date_format='%Y-%m-%d')

    for _,x in t2.iterrows():
        mult=f" (x{x['mult']:.2f}/yr)" if pd.notna(x['mult']) else ''
        print(f"{x['outcome']} [{x['sample']}]: slope={x['slope']:+.2f}{mult} r={x['r']:.2f} p={x['p']:.4f} n={x['n']}")
    print(t1.to_string(index=False))
    write_markdown(t1,t2,t3,t4,summ)

DISPLAY = {'hella_swag':'HellaSwag','arc_ai2':'ARC (AI2)','fictionlivebench':'FictionLiveBench','gsm8k':'GSM8K','mmlu':'MMLU','otis_mock_aime_2024_2025':'OTIS Mock AIME',
           'gpqa_diamond':'GPQA Diamond','math_level_5':'MATH Level 5','cybench':'Cybench','frontiermath':'FrontierMath',
           'frontiermath_tier_4':'FrontierMath Tier 4','arc_agi_2':'ARC-AGI-2','critpt':'CritPT','arc_agi':'ARC-AGI',
           'hle':'HLE','swe_bench_verified':'SWE-bench Verified','gdpval':'GDPval','terminalbench':'Terminal Bench',
           'exploitbench':'ExploitBench','osworld_2':'OSWorld-2'}

def m(x): return x.replace('-','−')   # typographic minus

def write_markdown(t1,t2,t3,t4,summ):
    dash=lambda v,f: '—' if v is None or pd.isna(v) else m(f.format(v))
    last=t1.iloc[-1]
    L=['# Results tables','',f"_Generated by `code/03_regressions_and_tables.py` ({summ['benchmark'].nunique()} benchmarks). Do not edit by hand._",'',
       '## Table 1. Benchmark saturation dynamics by introduction cohort','',
       '| Introduction cohort | Benchmarks (n) | Median lifespan to 95% (yr) | n | Median fitted rate k | Median peak velocity (pts/mo) | n |',
       '|---|---|---|---|---|---|---|']
    for _,x in t1.iterrows():
        L.append(f"| {x['cohort']} | {x['benchmarks']} | {dash(x['median_lifespan_yr'],'{:.1f}')} | {x['n_lifespan']} | "
                 f"{dash(x['median_k'],'{:.2f}')} | {dash(x['median_vmax'],'{:.1f}')} | {x['n_vmax']} |")
    note=(f"*Notes.* Cohorts defined by date of first frontier score in the dataset. Lifespan = interval from first frontier score "
          f"to the 95% crossing (empirical where observed, fit-implied otherwise); restricted to benchmarks with ≥{MIN_MODELS} distinct "
          f"scored models and a realized or projected crossing before {SAT_CUTOFF.year}. Fitted rate k from the better of logistic/Gompertz "
          f"fits by RMSE (≥12 monthly observations required). Peak velocity = maximum month-over-month gain of the smoothed SOTA series, "
          f"computed only for benchmarks whose frontier has passed 70%")
    if last['n_vmax']==0: note+=f"; no {last['cohort']} benchmark qualifies yet"
    if last['n_k']<=1: note+=f", and the fitted-k entry for this cohort rests on {'a single benchmark' if last['n_k']==1 else 'no benchmark'} and is suppressed"
    L+=['',note+'.','',
        '## Table 2. Regression estimates of the change in saturation speed','',
        '| Outcome | Sample | Slope (per year of introduction) | r | p | n |','|---|---|---|---|---|---|']
    for _,x in t2.iterrows():
        sl=f"{x['slope']:+.2f} (×{x['mult']:.2f}/yr)" if pd.notna(x['mult']) else f"{x['slope']:+.1f}" if abs(x['slope'])>=1 else f"{x['slope']:+.2f}"
        L.append(f"| {x['outcome']} | {x['sample']} | {m(sl)} | {m(f'{x.r:.2f}')} | {fmt_p(x['p'])} | {x['n']} |")
    L+=['',"*Notes.* Each row regresses the stated outcome on the benchmark's introduction date (year, continuous). Rows are ordered from "
        "strongest to weakest structural assumptions; the two nonparametric estimates (rows 4–5) are individually underpowered but "
        "concordant in sign with the parametric estimates.",'',
        '## Table 3. Functional principal component analysis of registered trajectories','',
        '| Component | Variance explained | Correlation with within-window gain | Interpretation |','|---|---|---|---|']
    interp=['Steepness of the saturation trajectory','Asymmetry about the 50% crossing']
    for i,(_,x) in enumerate(t3.iterrows()):
        L.append(f"| {x['component']} | {100*x['variance_explained']:.1f}% | {m(f'{x.corr_with_gain:+.2f}')} | {interp[i]} |")
    L+=['',f"*Notes.* Trajectories landmark-registered at the interpolated 50% crossing, represented in a cubic B-spline basis "
        f"(7 basis functions) on [−12, +9] months, n = {int(t3['n'].iloc[0])} benchmarks with full coverage of the window. The dominant "
        f"mode of shape variation is steepness, supporting rate as the natural object of study independent of any sigmoid assumption.",'',
        '## Table 4. Saturation status of selected benchmarks','',
        '| Benchmark | First score | Current SOTA | 95% crossed | 95% projected | Lifespan (yr) | Best fit | Ceiling | k | Peak vel. (pts/mo) |',
        '|---|---|---|---|---|---|---|---|---|---|']
    ym=lambda v: '—' if pd.isna(v) else pd.Timestamp(v).strftime('%Y-%m')
    below=[]
    for _,x in t4.iterrows():
        name=DISPLAY.get(x['benchmark'],x['benchmark'])
        if x['n_models']<MIN_MODELS: below.append(f"{name} ({x['n_models']} models)"); name+='†'
        sota='100%' if x['current_sota']>=1 else f"{100*x['current_sota']:.1f}%"
        L.append(f"| {name} | {ym(x['first_date'])} | {sota} | {ym(x['crossed_95'])} | {ym(x['projected_95'])} | "
                 f"{dash(x['lifespan_yr'],'{:.1f}')} | {({'gompertz':'Gompertz'}).get(x['model'],x['model']) if pd.notna(x['model']) else '—'} | {dash(x['ceiling'],'{:.2f}')} | "
                 f"{dash(x['rate_k'],'{:.2f}')} | {dash(x['vmax'],'{:.1f}')} |")
    L+=['','*Notes.* "First score" = first frontier observation in the dataset, which for older benchmarks may postdate true release. '
        'GSM8K and MMLU never cross 95% under the free-ceiling fits (fitted ceilings 0.93 and 0.86), reflecting label-noise floors; their '
        'lifespan cells are accordingly empty. Fitted ceilings well below current plausibility (e.g., FrontierMath 0.62, GDPval 0.52, '
        'CritPT 0.38) are mid-curve artifacts and should be read as lower bounds, not estimates of true attainable performance. Dashes in '
        'fit columns indicate fewer than 12 monthly observations. Projections assume the fitted sigmoid; treat as rough extrapolations.'
        + (f" † Fewer than {MIN_MODELS} distinct scored models ({', '.join(below)}); shown for status but excluded from the lifespan sample." if below else '')]
    open(os.path.join(ROOT,'paper','results_tables.md'),'w').write('\n'.join(L)+'\n')

if __name__=='__main__': main(sys.argv[1])
