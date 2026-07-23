"""SOTA frontiers, monthly panel, 50%-aligned curves, sigmoid fits, saturation summary.
Usage: python 02_saturation_analysis.py <all_benchmarks_long.csv> <output_dir>
"""
import pandas as pd, numpy as np, sys, os
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')

T0 = pd.Timestamp('2020-01-01')

def logistic_fixed(t,t0,k): return 1.0/(1.0+np.exp(-k*(t-t0)))          # ceiling = 1
def logistic3(t,L,t0,k):    return L/(1+np.exp(-k*(t-t0)))               # free ceiling
def gompertz3(t,L,t0,k):    return L*np.exp(-np.exp(-k*(t-t0)))

def yr(d): return (d - T0).dt.days/365.25 if hasattr(d,'dt') else (d-T0).days/365.25

def main(src, outdir):
    df = pd.read_csv(src, parse_dates=['date'])
    df = df[df['score']<=1.0]

    # --- frontiers & monthly panel ---
    panels, summ_rows = [], []
    for b,g in df.groupby('benchmark'):
        g=g.sort_values('date')
        s=g.set_index('date')['score'].cummax()
        s=s[~s.index.duplicated(keep='last')]
        m=s.resample('MS').max().ffill().reset_index()
        m.columns=['month','sota']; m['benchmark']=b
        panels.append(m)

        fr=m.rename(columns={'month':'date','sota':'sota'})
        t=yr(fr['date']).values; y=fr['sota'].values
        c90=fr.loc[fr['sota']>=0.90,'date'].min(); c95=fr.loc[fr['sota']>=0.95,'date'].min()
        proj95=None
        if len(fr)>=5 and 0.15<y.max()<0.95:
            try:
                p,_=curve_fit(logistic_fixed,t,y,p0=[t[-1],1.0],maxfev=20000)
                d95=T0+pd.Timedelta(days=(p[0]+np.log(0.95/0.05)/p[1])*365.25)
                if p[1]>0 and pd.Timestamp('2024-01-01')<d95<pd.Timestamp('2035-01-01'): proj95=d95
            except Exception: pass
        summ_rows.append(dict(benchmark=b,n_models=len(g),
            first_date=g['date'].min().date(),current_sota=round(y[-1],3),
            crossed_90=c90.date() if pd.notna(c90) else None,
            crossed_95=c95.date() if pd.notna(c95) else None,
            projected_95=proj95.date() if proj95 is not None else None))

    panel=pd.concat(panels)[['benchmark','month','sota']]
    panel.to_csv(f'{outdir}/sota_monthly_panel.csv',index=False)
    summ=pd.DataFrame(summ_rows)
    summ.to_csv(f'{outdir}/saturation_summary.csv',index=False)

    # --- 50%-aligned curves ---
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
    pd.concat(aligned)[['benchmark','month','months_since_50','sota']]\
      .to_csv(f'{outdir}/sota_aligned_at_50pct.csv',index=False)

    # --- free-ceiling logistic & Gompertz fits ---
    rows=[]
    for b,g in panel.groupby('benchmark'):
        if len(g)<12 or g['sota'].max()<0.3: continue
        t=yr(g['month']).values; y=g['sota'].values
        for name,f in [('logistic',logistic3),('gompertz',gompertz3)]:
            try:
                p,_=curve_fit(f,t,y,p0=[min(1.0,y.max()*1.05),t[len(t)//2],1.5],
                    bounds=([y.max()*0.9,-5,0.05],[1.0,20,20]),maxfev=50000)
                rmse=float(np.sqrt(np.mean((f(t,*p)-y)**2)))
                rows.append(dict(benchmark=b,model=name,ceiling=round(p[0],4),
                    midpoint_year=round(2020+p[1],2),rate_k=round(p[2],3),rmse=round(rmse,4),n=len(g)))
            except Exception: pass
    pd.DataFrame(rows).to_csv(f'{outdir}/saturation_curve_fits.csv',index=False)
    print("done")

if __name__=='__main__':
    os.makedirs(sys.argv[2],exist_ok=True); main(sys.argv[1],sys.argv[2])
