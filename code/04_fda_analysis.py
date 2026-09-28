"""Functional data analysis: landmark-registered FPCA of saturation trajectories.
Usage: python 04_fda_analysis.py <output_dir_from_step_02>
Requires: scikit-fda (pip install scikit-fda; with Python >= 3.14 also multimethod==1.12)
Writes table_fpca.csv and fpca_scores.csv; fpca() is also used by step 03 for the PC1 regression.
"""
import pandas as pd, numpy as np, sys
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

GRID = np.arange(-12, 10, 1.0)   # months relative to the 50% crossing, window [-12, +9]

def fpca(d):
    al=pd.read_csv(f'{d}/sota_aligned_at_50pct.csv')
    summ=pd.read_csv(f'{d}/saturation_summary.csv',parse_dates=['first_date'])
    curves,names=[],[]
    for b,g in al.groupby('benchmark'):
        g=g.sort_values('months_since_50')
        if g['months_since_50'].min()>-12 or g['months_since_50'].max()<9: continue
        curves.append(np.interp(GRID,g['months_since_50'],g['sota'])); names.append(b)
    X=np.array(curves)
    fd=skfda.FDataGrid(X,GRID).to_basis(BSplineBasis(domain_range=(-12,9),n_basis=7,order=4))
    model=FPCA(n_components=2); scores=model.fit_transform(fd)
    gain=X[:,-1]-X[:,0]
    corr=[np.corrcoef(scores[:,i],gain)[0,1] for i in range(2)]
    meta=summ.set_index('benchmark').loc[names]
    iy=(meta['first_date'].dt.year+meta['first_date'].dt.dayofyear/365).values
    return dict(names=names,X=X,scores=scores,evr=model.explained_variance_ratio_,corr=corr,iy=iy,
                n_aligned=al['benchmark'].nunique())

def main(d):
    r=fpca(d)
    print(f"04: {r['n_aligned']} benchmarks cross 50%; n fully observed = {len(r['names'])}: {r['names']}")
    # Eigenfunction signs are arbitrary; reported with the paper's convention
    # (PC1 = steepness, negative correlation with within-window gain; PC2 positive).
    pd.DataFrame({'component':['PC1','PC2'],
                  'variance_explained':np.round(r['evr'],4),
                  'corr_with_gain':[-abs(r['corr'][0]),abs(r['corr'][1])],
                  'interpretation':['Steepness of the saturation trajectory','Asymmetry about the 50% crossing'],
                  'n':len(r['names'])}).to_csv(f'{d}/table_fpca.csv',index=False)
    pd.DataFrame({'benchmark':r['names'],'intro_yr':r['iy'],'pc1':r['scores'][:,0],'pc2':r['scores'][:,1]})\
      .to_csv(f'{d}/fpca_scores.csv',index=False,float_format='%.6g')
    print("   explained variance:",np.round(r['evr'],3),"corr with gain:",np.round(r['corr'],2))

if __name__=='__main__': main(sys.argv[1])
