import pandas as pd, numpy as np, statsmodels.api as sm
fq = pd.read_parquet('ssa_firm_quarter.parquet')
fq['call_date'] = pd.to_datetime(fq['call_date'])
fq = fq.dropna(subset=['permno']).copy()
fq['call_qtr'] = fq['call_date'].dt.to_period('Q')
for v in ['ssa','credit_claiming','externalizing']:
    fq[f'{v}_z'] = (fq[v]-fq[v].mean())/fq[v].std()

fq = fq.sort_values(['permno','call_date'])
fq['next_sue'] = fq.groupby('permno')['sue'].shift(-1)
fq['next_call_date'] = fq.groupby('permno')['call_date'].shift(-1)
gap = (fq['next_call_date'] - fq['call_date']).dt.days
fq = fq[gap.between(60,150)].copy()

# Force float to avoid pandas nullable -> statsmodels crash
fq['sue'] = pd.to_numeric(fq['sue'], errors='coerce').astype(float)
fq['next_sue'] = pd.to_numeric(fq['next_sue'], errors='coerce').astype(float)
fq = fq.dropna(subset=['next_sue','sue']).copy()
print(f"After clean: {len(fq):,}")

for v in ['ssa_z','credit_claiming_z','externalizing_z']:
    coefs=[]
    for q,g in fq.groupby('call_qtr'):
        if len(g)<30: continue
        X = sm.add_constant(g[[v,'sue']].astype(float))
        y = g['next_sue'].astype(float)
        r = sm.OLS(y,X).fit()
        coefs.append(r.params)
    cdf = pd.DataFrame(coefs)
    s = cdf[v].dropna()
    nw = sm.OLS(s, np.ones(len(s))).fit(cov_type='HAC', cov_kwds={'maxlags':4})
    print(f"  {v:22s}: coef={s.mean():.5f}, t={nw.tvalues.iloc[0]:.2f}, N_qtr={len(s)}")

# Panel reg with firm FE — much higher power than FM with 20 quarters
print("\n=== Panel regression with quarter + firm FE ===")
import statsmodels.formula.api as smf

# Convert Period to string for patsy
fq['qtr_str'] = fq['call_qtr'].astype(str)
fq['permno_int'] = fq['permno'].astype(int)

for v in ['ssa_z','credit_claiming_z','externalizing_z']:
    fq['vv'] = fq[v]
    # Quarter FE absorbs aggregate time variation
    mod1 = smf.ols('next_sue ~ vv + sue + C(qtr_str)', data=fq).fit(
        cov_type='cluster', cov_kwds={'groups': fq['permno_int']})
    print(f"  [QtrFE only]  {v:22s}: coef={mod1.params['vv']:.5f}, "
          f"t={mod1.tvalues['vv']:.2f}, N={int(mod1.nobs):,}")
    
    # Quarter + firm FE (within-firm variation)
    mod2 = smf.ols('next_sue ~ vv + sue + C(qtr_str) + C(permno_int)', 
                   data=fq).fit(
        cov_type='cluster', cov_kwds={'groups': fq['permno_int']})
    print(f"  [Qtr+FirmFE] {v:22s}: coef={mod2.params['vv']:.5f}, "
          f"t={mod2.tvalues['vv']:.2f}, N={int(mod2.nobs):,}")
    
print("\n=== Externalizing robustness ===")
# Need to merge in size, BM, momentum from CRSP for controls
# Quick version: just current SUE + permno FE via FM with firm dummies
# For now, test if effect survives with stronger controls in panel

# Add current SSA components as controls (partial-out within-firm cycle)
for v in ['externalizing_z']:
    fq['vv'] = fq[v]
    # Lag-1 SUE control
    fq['lag_sue'] = fq.groupby('permno')['sue'].shift(1)
    mod = smf.ols('next_sue ~ vv + sue + lag_sue + credit_claiming_z + C(qtr_str) + C(permno_int)', 
                  data=fq.dropna(subset=['lag_sue'])).fit(
        cov_type='cluster', cov_kwds={'groups': fq.dropna(subset=['lag_sue'])['permno_int']})
    print(f"  Externalizing w/ lag SUE + credit_claiming controls: "
          f"coef={mod.params['vv']:.5f}, t={mod.tvalues['vv']:.2f}, N={int(mod.nobs):,}")
    
# 追加到 mechanism_debug.py
print("\n=== Horizon test: externalizing predicts SUE at h=1,2,3 ===")
print("Using within-transformation (firm + quarter demeaned)\n")

fq2 = pd.read_parquet('ssa_firm_quarter.parquet')
fq2['call_date'] = pd.to_datetime(fq2['call_date'])
fq2 = fq2.dropna(subset=['permno']).copy()
fq2['permno_int'] = fq2['permno'].astype(int)
fq2['qtr_str'] = fq2['call_date'].dt.to_period('Q').astype(str)
fq2['sue'] = pd.to_numeric(fq2['sue'], errors='coerce').astype(float)

for v in ['ssa','credit_claiming','externalizing']:
    fq2[f'{v}_z'] = (fq2[v]-fq2[v].mean())/fq2[v].std()

fq2 = fq2.sort_values(['permno_int','call_date'])
for h in [1,2,3]:
    fq2[f'sue_lead{h}'] = fq2.groupby('permno_int')['sue'].shift(-h)
for h in [1,2]:
    fq2[f'sue_lag{h}'] = fq2.groupby('permno_int')['sue'].shift(h)

def within_reg(data, yvar, xvars, cluster='permno_int'):
    """Two-way (firm + quarter) within-transformation, then OLS with cluster SE."""
    df = data.dropna(subset=[yvar] + xvars).copy()
    # Two-way demean: iterate until convergence (Gauss-Seidel)
    cols = [yvar] + xvars
    for c in cols:
        df[f'{c}_d'] = df[c].astype(float)
    for _ in range(20):
        # demean by firm
        for c in cols:
            df[f'{c}_d'] = df[f'{c}_d'] - df.groupby('permno_int')[f'{c}_d'].transform('mean')
        # demean by quarter
        for c in cols:
            df[f'{c}_d'] = df[f'{c}_d'] - df.groupby('qtr_str')[f'{c}_d'].transform('mean')
    X = df[[f'{x}_d' for x in xvars]].values
    y = df[f'{yvar}_d'].values
    # OLS with HC1, cluster by firm
    n, k = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    # Cluster-robust SE
    g = df[cluster].values
    XX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for cl in np.unique(g):
        idx = (g == cl)
        Xg = X[idx]
        eg = resid[idx]
        Xe = Xg.T @ eg
        meat += np.outer(Xe, Xe)
    n_cl = len(np.unique(g))
    dof_adj = (n_cl/(n_cl-1)) * ((n-1)/(n-k))
    vcov = dof_adj * XX_inv @ meat @ XX_inv
    se = np.sqrt(np.diag(vcov))
    tstat = beta / se
    return dict(zip(xvars, zip(beta, tstat))), int(n)

print("Forward horizons (h=1,2,3):")
for h in [1, 2, 3]:
    res, n = within_reg(fq2, f'sue_lead{h}', ['externalizing_z','sue'])
    coef, t = res['externalizing_z']
    print(f"  h=+{h}: coef={coef:.5f}, t={t:.2f}, N={n:,}")

print("\nFalsification (predicting past SUE):")
for h in [1, 2]:
    res, n = within_reg(fq2, f'sue_lag{h}', ['externalizing_z','sue'])
    coef, t = res['externalizing_z']
    print(f"  h=-{h}: coef={coef:.5f}, t={t:.2f}, N={n:,}")

print("\n=== Repeat for credit_claiming (for completeness) ===")
print("Forward horizons:")
for h in [1, 2, 3]:
    res, n = within_reg(fq2, f'sue_lead{h}', ['credit_claiming_z','sue'])
    coef, t = res['credit_claiming_z']
    print(f"  h=+{h}: coef={coef:.5f}, t={t:.2f}, N={n:,}")

# 加在 mechanism_debug.py 末尾
print("\n=== Robustness: control for LLM-coded certainty (vague-language proxy) ===")

# fq2 already has pct_low_certainty and pct_deflective from ssa_firm_quarter.parquet
print(f"fq2 columns: {fq2.columns.tolist()}")
print(f"\npct_low_certainty stats: mean={fq2['pct_low_certainty'].mean():.3f}, "
      f"non-null={fq2['pct_low_certainty'].notna().sum():,}")
print(f"pct_deflective stats:    mean={fq2['pct_deflective'].mean():.3f}, "
      f"non-null={fq2['pct_deflective'].notna().sum():,}")

# Standardize for interpretability
for c in ['pct_low_certainty', 'pct_deflective']:
    fq2[f'{c}_z'] = (fq2[c] - fq2[c].mean()) / fq2[c].std()

print("\n--- Externalizing controlling for low_certainty + deflective ---")
print("(within-firm + within-quarter transformation)")
for h in [1, 2]:
    res, n = within_reg(fq2, f'sue_lead{h}', 
                        ['externalizing_z', 'sue', 
                         'pct_low_certainty_z', 'pct_deflective_z'])
    coef, t = res['externalizing_z']
    cer_coef, cer_t = res['pct_low_certainty_z']
    def_coef, def_t = res['pct_deflective_z']
    print(f"  h=+{h}:")
    print(f"    externalizing_z:    coef={coef:+.5f}, t={t:+.2f}")
    print(f"    pct_low_certainty:  coef={cer_coef:+.5f}, t={cer_t:+.2f}")
    print(f"    pct_deflective:     coef={def_coef:+.5f}, t={def_t:+.2f}")
    print(f"    N={n:,}")

# Also: control for credit_claiming jointly (the strongest test)
print("\n--- Externalizing controlling for credit_claiming + certainty + deflective ---")
for h in [1, 2]:
    res, n = within_reg(fq2, f'sue_lead{h}', 
                        ['externalizing_z', 'credit_claiming_z', 'sue',
                         'pct_low_certainty_z', 'pct_deflective_z'])
    coef, t = res['externalizing_z']
    cc_coef, cc_t = res['credit_claiming_z']
    print(f"  h=+{h}:")
    print(f"    externalizing_z:    coef={coef:+.5f}, t={t:+.2f}")
    print(f"    credit_claiming_z:  coef={cc_coef:+.5f}, t={cc_t:+.2f}")
    print(f"    N={n:,}")