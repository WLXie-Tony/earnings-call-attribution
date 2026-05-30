"""
Asset pricing tests: portfolio sorts + Fama-MacBeth on SSA.
Tests whether firm-quarter SSA predicts subsequent abnormal returns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# ===== Load data =====
fq = pd.read_parquet('ssa_firm_quarter.parquet')
fq['call_date'] = pd.to_datetime(fq['call_date'])
print(f"Before dropping NA permno: {len(fq):,}")
fq = fq.dropna(subset=['permno']).copy()
fq['permno'] = fq['permno'].astype(int)
print(f"After dropping NA permno:  {len(fq):,}")

crsp = pd.read_parquet('crsp_returns.parquet')
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['permno'] = crsp['permno'].astype(int)
crsp = crsp.sort_values(['permno', 'date'])

ff_m = pd.read_parquet('ff_factors_monthly.parquet')
ff_m['date'] = pd.to_datetime(ff_m['date'])
ff_m['ym'] = ff_m['date'].dt.to_period('M')

print(f"Firm-quarters: {len(fq):,}")

# ===== Build forward returns for each call =====
# For each (permno, call_date), compute cumulative return over months t+1 to t+k
def forward_return(permno, call_date, months):
    """Cumulative return starting the month AFTER the call."""
    start = (call_date + pd.offsets.MonthBegin(1))
    end = (start + pd.offsets.MonthEnd(months))
    sub = crsp[(crsp['permno']==permno) & (crsp['date']>=start) & (crsp['date']<=end)]
    if len(sub) < months * 15:  # require minimum trading days
        return np.nan
    return (1 + sub['ret'].fillna(0)).prod() - 1

print("\nComputing forward returns (this takes a few minutes)...")
# Vectorized approach: pre-index crsp by permno
crsp_by_permno = {p: g.set_index('date')['ret'] for p, g in crsp.groupby('permno')}

def fwd_ret_fast(row, months):
    s = crsp_by_permno.get(row['permno'])
    if s is None:
        return np.nan
    start = row['call_date'] + pd.offsets.MonthBegin(1)
    end = start + pd.offsets.MonthEnd(months)
    window = s[(s.index >= start) & (s.index <= end)]
    if len(window) < months * 15:
        return np.nan
    return (1 + window.fillna(0)).prod() - 1

for k in [1, 3, 6]:
    fq[f'fwd_ret_{k}m'] = fq.apply(lambda r: fwd_ret_fast(r, k), axis=1)
    print(f"  fwd_ret_{k}m: {fq[f'fwd_ret_{k}m'].notna().sum():,} non-missing")

# ===== Portfolio sort on SSA =====
print("\n=== Portfolio sort on SSA (quintiles) ===")
fq['call_qtr'] = fq['call_date'].dt.to_period('Q')
# Quintile within each quarter (avoid look-ahead)
fq['ssa_quintile'] = fq.groupby('call_qtr')['ssa'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=[1,2,3,4,5])
)

for k in [1, 3, 6]:
    print(f"\n--- Holding period: {k} months ---")
    port = fq.groupby('ssa_quintile', observed=True)[f'fwd_ret_{k}m'].agg(['mean','count'])
    port['mean_pct'] = port['mean'] * 100
    print(port[['mean_pct','count']])
    q5 = fq[fq['ssa_quintile']==5][f'fwd_ret_{k}m'].mean()
    q1 = fq[fq['ssa_quintile']==1][f'fwd_ret_{k}m'].mean()
    print(f"  Q5-Q1 spread: {(q5-q1)*100:.3f}% over {k} months")

# ===== Fama-MacBeth regression =====
print("\n=== Fama-MacBeth (monthly cross-sectional) ===")
# Assign each call to its quarter, run cross-sectional reg of fwd_ret on SSA + controls
# Simplified: quarter-by-quarter cross-sectional regression
def fama_macbeth(data, yvar, xvars):
    coefs = []
    for q, g in data.groupby('call_qtr'):
        g = g.dropna(subset=[yvar] + xvars)
        if len(g) < 30:
            continue
        X = sm.add_constant(g[xvars])
        y = g[yvar]
        try:
            res = sm.OLS(y, X).fit()
            coefs.append(res.params)
        except:
            continue
    coef_df = pd.DataFrame(coefs)
    # Newey-West on the time series of coefficients
    results = {}
    for c in coef_df.columns:
        series = coef_df[c].dropna()
        mean = series.mean()
        nw = sm.OLS(series, np.ones(len(series))).fit(
            cov_type='HAC', cov_kwds={'maxlags': 4})
        results[c] = (mean, nw.tvalues[0])
    return results, len(coef_df)

# Standardize SSA for interpretability
fq['ssa_z'] = (fq['ssa'] - fq['ssa'].mean()) / fq['ssa'].std()

fm_results, n_qtrs = fama_macbeth(fq, 'fwd_ret_6m', ['ssa_z'])
print(f"Cross-sections (quarters): {n_qtrs}")
for var, (coef, tstat) in fm_results.items():
    print(f"  {var:12s}: coef={coef:.5f}, t={tstat:.2f}")

fq.to_parquet('ssa_with_returns.parquet', index=False)
print("\nSaved ssa_with_returns.parquet")

# ===== Component analysis: credit_claiming vs externalizing =====
print("\n" + "="*60)
print("COMPONENT ANALYSIS: credit_claiming vs externalizing")
print("="*60)

for var in ['credit_claiming', 'externalizing', 'ssa']:
    fq[f'{var}_z'] = (fq[var] - fq[var].mean()) / fq[var].std()

# Portfolio sort on each component separately
for var in ['credit_claiming', 'externalizing']:
    print(f"\n--- Quintile sort on {var}, 6-month holding ---")
    fq[f'{var}_q'] = fq.groupby('call_qtr')[var].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=[1,2,3,4,5])
    )
    port = fq.groupby(f'{var}_q', observed=True)['fwd_ret_6m'].agg(['mean', 'count'])
    port['mean_pct'] = port['mean'] * 100
    print(port[['mean_pct', 'count']])
    q5 = fq[fq[f'{var}_q']==5]['fwd_ret_6m'].mean()
    q1 = fq[fq[f'{var}_q']==1]['fwd_ret_6m'].mean()
    print(f"  Q5-Q1 spread: {(q5-q1)*100:.3f}%")

# Fama-MacBeth on each component (and jointly)
print("\n--- Fama-MacBeth, fwd_ret_6m ---")
for spec_name, xvars in [
    ('Univariate: credit_claiming',  ['credit_claiming_z']),
    ('Univariate: externalizing',    ['externalizing_z']),
    ('Univariate: ssa',              ['ssa_z']),
    ('Joint: cc + ext',              ['credit_claiming_z', 'externalizing_z']),
]:
    fm, n = fama_macbeth(fq, 'fwd_ret_6m', xvars)
    print(f"\n  [{spec_name}]  N quarters = {n}")
    for v, (c, t) in fm.items():
        print(f"    {v:25s}: coef={c:.5f}, t={t:.2f}")