"""SOTA frontiers, saturation summary, record-frontier monthly panel, sigmoid fits,
50%-aligned curves.
Usage: python 02_saturation_analysis.py <all_benchmarks_long.csv> <output_dir>

Method (as published): 95% crossings and first-score dates are read off the frontier on the
raw release-date axis; where a benchmark has not crossed, a fixed-ceiling logistic fit on that
frontier projects the crossing. The monthly panel used for curve fitting and registration holds
record-setting scores only (strictly increasing running maximum by release date), spans each
benchmark's first to last record date, and is resampled month-start and forward-filled.
"""
import pandas as pd, numpy as np, sys, os
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')
from sat_config import T0

def logistic_fixed(t,t0,k): return 1.0/(1.0+np.exp(-k*(t-t0)))          # ceiling = 1
def logistic3(t,L,t0,k):    return L/(1+np.exp(-k*(t-t0)))               # free ceiling
def gompertz3(t,L,t0,k):    return L*np.exp(-np.exp(-k*(t-t0)))

def yr(d): return (d - T0).dt.days/365.25

def summary(df):
    rows=[]
    for b,g in df.groupby('benchmark'):
        g=g.sort_values('date')
        g2=g.copy(); g2['sota']=g2['score'].cummax()
        fr=g2.groupby('date',as_index=False)['sota'].max()
        t=yr(fr['date']).values; y=fr['sota'].values
        c90=fr.loc[fr['sota']>=0.90,'date'].min(); c95=fr.loc[fr['sota']>=0.95,'date'].min()
        proj=None
        if len(fr)>=5 and 0.15<y.max()<0.95:
            try:
                p,_=curve_fit(logistic_fixed,t,y,p0=[t[-1],1.0],maxfev=20000)
                d95=T0+pd.Timedelta(days=(p[0]+np.log(0.95/0.05)/p[1])*365.25)
                if p[1]>0 and pd.Timestamp('2024-01-01')<d95<pd.Timestamp('2035-01-01'): proj=d95
            except Exception: pass
        rows.append(dict(benchmark=b,n_models=g['model'].nunique(),n_rows=len(g),
            first_date=g['date'].min().date(),last_date=g['date'].max().date(),current_sota=round(y[-1],3),
            crossed_90=c90.date() if pd.notna(c90) else None,
            crossed_95=c95.date() if pd.notna(c95) else None,
            projected_95=proj.date() if proj is not None else None))
    return pd.DataFrame(rows)

def record_panel(df):
    panels=[]
    for b,g in df.groupby('benchmark'):
        g=g.sort_values(['date','score'])
        rec,best=[],-1.0
        for d,s in zip(g['date'],g['score']):
            if s>best: best=s; rec.append((d,s))
        fr=pd.DataFrame(rec,columns=['date','score'])
        s=fr.set_index('date')['score'].cummax()
        s=s[~s.index.duplicated(keep='last')]
        m=s.resample('MS').max().ffill().reset_index()
        m.columns=['month','sota']; m['benchmark']=b
        panels.append(m)
    return pd.concat(panels)[['benchmark','month','sota']].reset_index(drop=True)

def aligned_at_50(panel):
    aligned=[]
    for b,g in panel.groupby('benchmark'):
        g=g.sort_values('month'); y=g['sota'].values
        idx=np.argmax(y>=0.5)
        if y[idx]<0.5 or len(g)<10: continue
        if idx==0: c50=g['month'].iloc[0]
        else:
            y0,y1=y[idx-1],y[idx]; t0,t1=g['month'].iloc[idx-1],g['month'].iloc[idx]
            c50=t0+(t1-t0)*((0.5-y0)/(y1-y0) if y1>y0 else 0)
        gg=g.copy(); gg['months_since_50']=((gg['month']-c50).dt.days/30.44).round(2)
        aligned.append(gg)
    return pd.concat(aligned)[['benchmark','month','months_since_50','sota']]

def curve_fits(panel):
    rows=[]
    for b,g in panel.groupby('benchmark'):
        if len(g)<12 or g['sota'].max()<0.3: continue
        t=yr(g['month']).values; y=g['sota'].values
        for name,f in [('logistic',logistic3),('gompertz',gompertz3)]:
            try:
                p,_=curve_fit(f,t,y,p0=[min(1.0,y.max()*1.05),t[len(t)//2],1.5],
                    bounds=([y.max()*0.9,-5,0.05],[1.0,20,20]),maxfev=50000)
                rmse=float(np.sqrt(np.mean((f(t,*p)-y)**2)))
                rows.append(dict(benchmark=b,model=name,ceiling=p[0],midpoint_year=2020+p[1],
                                 rate_k=p[2],rmse=rmse,n=len(g)))
            except Exception: pass
    return pd.DataFrame(rows)

def main(src, outdir):
    df=pd.read_csv(src,parse_dates=['date'])
    df=df[df['score']<=1.0]
    summary(df).to_csv(f'{outdir}/saturation_summary.csv',index=False)
    panel=record_panel(df)
    panel.to_csv(f'{outdir}/sota_monthly_panel.csv',index=False)
    aligned_at_50(panel).to_csv(f'{outdir}/sota_aligned_at_50pct.csv',index=False)
    curve_fits(panel).to_csv(f'{outdir}/saturation_curve_fits.csv',index=False,float_format='%.6g')
    print(f"02: {df['benchmark'].nunique()} benchmarks, {len(panel)} benchmark-months")

if __name__=='__main__':
    os.makedirs(sys.argv[2],exist_ok=True); main(sys.argv[1],sys.argv[2])
