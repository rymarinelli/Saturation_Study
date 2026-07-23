"""Cohort statistics and regression estimates (Tables 1-2 of the paper).
Usage: python 03_regressions_and_tables.py <output_dir_from_step_02>
"""
import pandas as pd, numpy as np, sys
from scipy import stats
from scipy.ndimage import uniform_filter1d

def main(d):
    summ=pd.read_csv(f'{d}/saturation_summary.csv',parse_dates=['crossed_95','projected_95','first_date'])
    panel=pd.read_csv(f'{d}/sota_monthly_panel.csv',parse_dates=['month'])

    s=summ.copy()
    s['sat']=s['crossed_95'].fillna(s['projected_95'])
    s=s[pd.notna(s['sat'])&(s['n_models']>=8)&(s['sat']<pd.Timestamp('2030-01-01'))]
    s['months']=(s['sat']-s['first_date']).dt.days/30.44
    s=s[s['months']>0]
    x=(s['first_date']-pd.Timestamp('2020-01-01')).dt.days/365.25
    r=stats.linregress(x,s['months'])
    print(f"Lifespan~intro (all, n={len(s)}): slope={r.slope:.1f} mo/yr r={r.rvalue:.2f} p={r.pvalue:.4f}")
    e=s[pd.notna(s['crossed_95'])]
    xe=(e['first_date']-pd.Timestamp('2020-01-01')).dt.days/365.25
    r=stats.linregress(xe,e['months'])
    print(f"Lifespan~intro (empirical, n={len(e)}): slope={r.slope:.1f} mo/yr r={r.rvalue:.2f} p={r.pvalue:.4f}")

    fits=pd.read_csv(f'{d}/saturation_curve_fits.csv')
    best=fits.sort_values('rmse').groupby('benchmark').first().reset_index()
    best=best.merge(summ[['benchmark','first_date']],on='benchmark')
    best['iy']=best['first_date'].dt.year+best['first_date'].dt.dayofyear/365
    b=best[best['rate_k']<15]
    r=stats.linregress(b['iy'],np.log(b['rate_k']))
    print(f"log(k)~intro (n={len(b)}): slope={r.slope:.2f} (x{np.exp(r.slope):.2f}/yr) r={r.rvalue:.2f} p={r.pvalue:.4f}")

    rows=[]
    for bch,g in panel.groupby('benchmark'):
        g=g.sort_values('month')
        if len(g)<12 or g['sota'].max()<0.70: continue
        ys=uniform_filter1d(g['sota'].values,size=3)
        fr=summ.set_index('benchmark').loc[bch]
        rows.append((fr['first_date'].year+fr['first_date'].dayofyear/365, np.diff(ys).max()*100))
    v=pd.DataFrame(rows,columns=['iy','vmax'])
    r=stats.linregress(v['iy'],np.log(v['vmax'].clip(lower=0.1)))
    print(f"log(vmax)~intro (n={len(v)}): slope={r.slope:.3f} (x{np.exp(r.slope):.2f}/yr) r={r.rvalue:.2f} p={r.pvalue:.4f}")

    for lo,hi in [(2019,2023),(2023,2024),(2024,2025),(2025,2027)]:
        sel=s[(s['first_date']>=f'{lo}-01-01')&(s['first_date']<f'{hi}-01-01')]
        vv=v[(v['iy']>=lo)&(v['iy']<hi)]
        kk=b[(b['iy']>=lo)&(b['iy']<hi)]
        print(f"cohort {lo}-{hi-1}: lifespan med {sel['months'].median()/12:.1f}yr (n={len(sel)}), "
              f"k med {kk['rate_k'].median():.2f} (n={len(kk)}), vmax med {vv['vmax'].median():.1f} (n={len(vv)})")

if __name__=='__main__': main(sys.argv[1])
