"""
Mechanism: does SSA predict future operating performance?
Test on next-quarter SUE, ROA, analyst revisions.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

fq = pd.read_parquet('ssa_firm_quarter.parquet')
fq['call_date'] = pd.to_datetime(fq['call_date'])
fq = fq.dropna(subset=['permno']).copy()
fq['permno'] = fq['permno'].astype(int)
fq['call_qtr'] = fq['call_date'].dt.to_period('Q')

# Standardize SSA and components
for v in ['ssa', 'credit_claiming', 'externalizing']:
    fq[f'{v}_z'] = (fq[v] - fq[v].mean()) / fq[v].std()

# === 1. Next-quarter SUE ===
print("=== Next-quarter SUE prediction ===")
# Sort by (permno, call_date), get NEXT call's SUE for same firm
fq_sorted = fq.sort_values(['permno', 'call_date'])
fq_sorted['next_sue'] = fq_sorted.groupby('permno')['sue'].shift(-1)
fq_sorted['next_call_date'] = fq_sorted.groupby('permno')['call_date'].shift(-1)
gap = (fq_sorted['next_call_date'] - fq_sorted['call_date']).dt.days
fq_sorted = fq_sorted[gap.between(60, 150)].copy()  # next call within 2-5 months
print(f"Firm-quarters with valid next-quarter SUE: {len(fq_sorted):,}")

for v in ['ssa_z', 'credit_claiming_z', 'externalizing_z']:
    coefs = []
    for q, g in fq_sorted.dropna(subset=['next_sue', v]).groupby('call_qtr'):
        if len(g) < 30: continue
        X = sm.add_constant(g[[v, 'sue']])  # control for current SUE
        y = g['next_sue']
        try: coefs.append(sm.OLS(y, X).fit().params)
        except: continue
    cdf = pd.DataFrame(coefs)
    if v in cdf.columns:
        s = cdf[v].dropna()
        nw = sm.OLS(s, np.ones(len(s))).fit(cov_type='HAC', cov_kwds={'maxlags':4})
        print(f"  {v:25s}: coef={s.mean():.5f}, t={nw.tvalues.iloc[0]:.2f}, N_qtr={len(s)}")

# === 2. Next-quarter ROA via Compustat ===
print("\n=== Next-quarter ROA prediction ===")
comp = pd.read_parquet('compustat_fundq.parquet')
comp['datadate'] = pd.to_datetime(comp['datadate'])

# Need permno-gvkey link via CCM (skip for now — use ticker-based merge if needed)
# Simplified: skip ROA if no permno-gvkey link readily available
# We'll try a CRSP-CCM link
print("  (ROA test requires permno-gvkey link via CCM — see notes)")

# === 3. Analyst forecast revisions post-call ===
print("\n=== Analyst forecast revisions ===")
rev = pd.read_parquet('ibes_revisions.parquet')
rev['statpers'] = pd.to_datetime(rev['statpers'])
# need ticker linked to permno — load sue_panel_final which has both
sue_panel = pd.read_parquet('sue_panel_final.parquet')
fq_with_ticker = fq.merge(sue_panel[['companyid','call_date','ticker']], on=['companyid','call_date'], how='left')

def post_call_revision(row):
    """Mean forecast revision from pre-call to (call+90d)."""
    t = row['ticker']
    d = row['call_date']
    pre = rev[(rev['ticker']==t) & (rev['statpers'].between(d - pd.Timedelta(30,'d'), d))]
    post = rev[(rev['ticker']==t) & (rev['statpers'].between(d + pd.Timedelta(7,'d'), d + pd.Timedelta(90,'d')))]
    if len(pre)==0 or len(post)==0: return np.nan
    # focus on same fpedats (next quarter forecast)
    pre_med = pre.groupby('fpedats')['medest'].mean()
    post_med = post.groupby('fpedats')['medest'].mean()
    common = pre_med.index.intersection(post_med.index)
    if len(common)==0: return np.nan
    return (post_med[common] - pre_med[common]).mean()

print("Computing post-call forecast revisions (slow, ~5-10 min)...")
fq_with_ticker['fcst_revision'] = fq_with_ticker.apply(post_call_revision, axis=1)
print(f"Non-missing revisions: {fq_with_ticker['fcst_revision'].notna().sum():,}")

for v in ['ssa_z', 'credit_claiming_z', 'externalizing_z']:
    sub = fq_with_ticker.dropna(subset=['fcst_revision', v]).copy()
    sub[v] = (sub[v] - sub[v].mean()) / sub[v].std()
    coefs = []
    for q, g in sub.groupby('call_qtr'):
        if len(g) < 30: continue
        X = sm.add_constant(g[[v]])
        y = g['fcst_revision']
        try: coefs.append(sm.OLS(y, X).fit().params)
        except: continue
    cdf = pd.DataFrame(coefs)
    if v in cdf.columns:
        s = cdf[v].dropna()
        if len(s) > 0:
            nw = sm.OLS(s, np.ones(len(s))).fit(cov_type='HAC', cov_kwds={'maxlags':4})
            print(f"  {v:25s}: coef={s.mean():.5f}, t={nw.tvalues.iloc[0]:.2f}, N_qtr={len(s)}")