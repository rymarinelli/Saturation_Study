"""Functional data analysis: landmark-registered FPCA + peak velocity.
Usage: python 04_fda_analysis.py <output_dir_from_step_02>
Requires: scikit-fda (pip install scikit-fda)
"""
import pandas as pd, numpy as np, sys
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

def main(d):
    al=pd.read_csv(f'{d}/sota_aligned_at_50pct.csv')
    summ=pd.read_csv(f'{d}/saturation_summary.csv',parse_dates=['first_date'])
    grid=np.arange(-12,10,1.0)
    curves,names=[],[]
    for b,g in al.groupby('benchmark'):
        g=g.sort_values('months_since_50')
        if g['months_since_50'].min()>-12 or g['months_since_50'].max()<9: continue
        curves.append(np.interp(grid,g['months_since_50'],g['sota'])); names.append(b)
    X=np.array(curves)
    print(f"n fully observed = {len(names)}: {names}")
    fd=skfda.FDataGrid(X,grid).to_basis(BSplineBasis(domain_range=(-12,9),n_basis=7,order=4))
    fpca=FPCA(n_components=2); scores=fpca.fit_transform(fd)
    print("explained variance:",np.round(fpca.explained_variance_ratio_,3))
    gain=X[:,-1]-X[:,0]
    for i in range(2): print(f"corr(PC{i+1},gain)={np.corrcoef(scores[:,i],gain)[0,1]:.2f}")
    meta=summ.set_index('benchmark').loc[names]
    iy=(meta['first_date'].dt.year+meta['first_date'].dt.dayofyear/365).values
    for i in range(2):
        r=stats.linregress(iy,scores[:,i])
        print(f"PC{i+1}~intro: slope={r.slope:.3f} r={r.rvalue:.2f} p={r.pvalue:.4f}")

if __name__=='__main__': main(sys.argv[1])
