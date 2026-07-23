"""All figures: saturation grid, timeline, styled per-benchmark plots, time-to-saturation
Gantt, and 50%-aligned overlay. Usage: python 05_figures.py <data_dir> <fig_dir>
Chart style matches the model-card figures: orange (#C96A3B) lines with labeled points,
dashed gray 100% saturation line, bold titles.
"""
import pandas as pd, numpy as np, sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import warnings; warnings.filterwarnings('ignore')

ORANGE='#C96A3B'; GREEN='#1B9E77'; T0=pd.Timestamp('2020-01-01')
def logistic(t,t0,k): return 1.0/(1.0+np.exp(-k*(t-t0)))

def frontier(panel,b):
    g=panel[panel.benchmark==b].sort_values('month')
    return g['month'].values, g['sota'].values

def main(d, figdir):
    panel=pd.read_csv(f'{d}/sota_monthly_panel.csv',parse_dates=['month'])
    summ=pd.read_csv(f'{d}/saturation_summary.csv',parse_dates=['crossed_95','projected_95','first_date'])
    al=pd.read_csv(f'{d}/sota_aligned_at_50pct.csv')

    # --- time-to-saturation Gantt ---
    s=summ.copy(); s['sat']=s['crossed_95'].fillna(s['projected_95'])
    s=s[pd.notna(s['sat'])&(s['n_models']>=8)&(s['sat']<pd.Timestamp('2029-07-01'))]
    s['start']=s['first_date']; s['months']=(s['sat']-s['start']).dt.days/30.44
    s=s[s['months']>0].sort_values('start')
    fig,ax=plt.subplots(figsize=(15,0.45*len(s)+2))
    for i,(_,r) in enumerate(s.iterrows()):
        emp=pd.notna(r['crossed_95'])
        ax.plot([r['start'],r['sat']],[i,i],lw=6,color=ORANGE,alpha=0.95 if emp else 0.35,solid_capstyle='butt')
        ax.plot(r['sat'],i,'o',ms=8,color=ORANGE if emp else 'white',mec=ORANGE,mew=1.8)
        ax.text(r['sat'],i,f"  {r['benchmark'].replace('_',' ')} — {r['months']/12:.1f} yr"+('' if emp else ' (proj.)'),va='center',fontsize=10.5)
    ax.axvline(pd.Timestamp.today(),color='gray',ls='--',lw=1.2)
    ax.set_yticks([]); ax.set_ylim(-1,len(s)+0.5)
    ax.set_title('How long until a benchmark saturates?  (bar = first frontier score → SOTA ≥95 %; faded = projected)',fontsize=15,fontweight='bold')
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout(); plt.savefig(f'{figdir}/time_to_saturation.png',dpi=120,bbox_inches='tight'); plt.close()

    # --- aligned overlay (median truncated to offsets with >=10 benchmarks) ---
    recent=al.groupby('benchmark')['month'].min().pipe(lambda x:x[pd.to_datetime(x)>='2024-01-01']).index
    fig,ax=plt.subplots(figsize=(14,8))
    ax.axhline(100,color='gray',ls='--',lw=1.2,alpha=0.7); ax.axvline(0,color='gray',lw=0.8,alpha=0.5)
    for b,g in al.groupby('benchmark'):
        g=g.sort_values('months_since_50')
        ax.plot(g['months_since_50'],g['sota']*100,'-',color=GREEN if b in recent else ORANGE,alpha=0.45,lw=1.4)
    al2=al.copy(); al2['bin']=(al2['months_since_50']/3).round()*3
    counts=al2.groupby('bin')['benchmark'].nunique()
    med=al2.groupby('bin')['sota'].median()
    keep=counts[counts>=10].index
    med=med[med.index.isin(keep)]
    ax.plot(med.index,med.values*100,color='black',lw=3.5,label='Median (≥10 benchmarks per offset)')
    ax.set_xlim(-40,30); ax.set_ylim(0,112)
    ax.set_xlabel('Months since SOTA crossed 50 %'); ax.set_ylabel('SOTA (%)')
    ax.set_title('Benchmark saturation curves aligned at the 50 % crossing\n(orange = first scored pre-2024, green = 2024 onward)',fontsize=15,fontweight='bold')
    ax.legend(fontsize=12,loc='lower right')
    plt.tight_layout(); plt.savefig(f'{figdir}/aligned_saturation_curves.png',dpi=130,bbox_inches='tight'); plt.close()
    print("figures written to",figdir)

if __name__=='__main__':
    os.makedirs(sys.argv[2],exist_ok=True); main(sys.argv[1],sys.argv[2])
