"""
Pull CRSP returns + IBES revisions for asset pricing and mechanism tests.

Outputs:
  crsp_returns.parquet  - daily returns around each call (-5 to +126 trading days)
  ff_factors.parquet    - Fama-French 5 factors + momentum (daily + monthly)
  ibes_revisions.parquet - analyst forecast revisions around each call
  future_fundamentals.parquet - next-quarter SUE, ROA for mechanism test
"""
import wrds
import pandas as pd
import numpy as np
from pathlib import Path

db = wrds.Connection()

# Load our firm-quarter panel
fq = pd.read_parquet('ssa_firm_quarter.parquet')
fq['call_date'] = pd.to_datetime(fq['call_date'])
permnos = fq['permno'].dropna().astype(int).unique().tolist()
print(f"Firm-quarter events: {len(fq):,}, unique permnos: {len(permnos):,}")

# ========== 1. CRSP daily returns ==========
print("\n[1] Pulling CRSP daily returns...")
permno_str = ','.join(str(p) for p in permnos)
crsp_chunks = []
for i in range(0, len(permnos), 1000):
    batch = permnos[i:i+1000]
    bstr = ','.join(str(p) for p in batch)
    q = f"""
    SELECT permno, date, ret, prc, shrout, vol
    FROM crsp.dsf
    WHERE permno IN ({bstr})
      AND date >= '2018-10-01' AND date <= '2024-09-30'
    """
    crsp_chunks.append(db.raw_sql(q))
    print(f"  batch {i//1000+1}/{(len(permnos)+999)//1000}: {len(crsp_chunks[-1]):,} rows")
crsp = pd.concat(crsp_chunks, ignore_index=True)
crsp['date'] = pd.to_datetime(crsp['date'])
crsp.to_parquet('crsp_returns.parquet', index=False)
print(f"  Saved {len(crsp):,} daily return obs")

# ========== 2. Fama-French factors ==========
print("\n[2] Pulling FF5 + momentum...")
ff_daily = db.raw_sql("""
    SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
    FROM ff.fivefactors_daily
    WHERE date >= '2018-10-01' AND date <= '2024-09-30'
""")
ff_daily['date'] = pd.to_datetime(ff_daily['date'])
ff_daily.to_parquet('ff_factors_daily.parquet', index=False)

ff_monthly = db.raw_sql("""
    SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
    FROM ff.fivefactors_monthly
    WHERE date >= '2018-10-01' AND date <= '2024-09-30'
""")
ff_monthly['date'] = pd.to_datetime(ff_monthly['date'])
ff_monthly.to_parquet('ff_factors_monthly.parquet', index=False)
print(f"  Saved FF factors (daily {len(ff_daily)}, monthly {len(ff_monthly)})")

# ========== 3. IBES forecast revisions (mechanism) ==========
print("\n[3] Pulling IBES revisions...")
# Get tickers from our IBES link (need to re-derive or store earlier)
# We'll pull statsum around each call to measure post-call revisions
cusip8 = fq.merge(
    pd.read_parquet('sue_panel_final.parquet')[['companyid','call_date','cusip','ticker']],
    on=['companyid','call_date'], how='left'
)
tickers = cusip8['ticker'].dropna().unique().tolist()
print(f"  Unique IBES tickers: {len(tickers):,}")
rev_chunks = []
for i in range(0, len(tickers), 1000):
    batch = tickers[i:i+1000]
    bstr = ','.join(f"'{t}'" for t in batch)
    q = f"""
    SELECT ticker, statpers, fpedats, fpi, meanest, medest, numest, stdev
    FROM ibes.statsum_epsus
    WHERE ticker IN ({bstr})
      AND fpi IN ('1','6')
      AND statpers >= '2018-10-01' AND statpers <= '2024-06-30'
      AND measure = 'EPS'
    """
    rev_chunks.append(db.raw_sql(q))
    print(f"  batch {i//1000+1}/{(len(tickers)+999)//1000}: {len(rev_chunks[-1]):,} rows")
rev = pd.concat(rev_chunks, ignore_index=True)
rev['statpers'] = pd.to_datetime(rev['statpers'])
rev['fpedats'] = pd.to_datetime(rev['fpedats'])
rev.to_parquet('ibes_revisions.parquet', index=False)
print(f"  Saved {len(rev):,} forecast snapshots")

# ========== 4. Future fundamentals (next-quarter ROA via Compustat) ==========
print("\n[4] Pulling Compustat quarterly fundamentals...")
gvkey_link = pd.read_parquet('sue_panel_final.parquet') if Path('sue_panel_final.parquet').exists() else None
# Pull quarterly ROA, earnings for mechanism test
q = f"""
SELECT gvkey, datadate, rdq, niq, atq, saleq, oiadpq
FROM comp.fundq
WHERE datadate >= '2018-10-01' AND datadate <= '2024-09-30'
  AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
"""
comp = db.raw_sql(q)
comp['datadate'] = pd.to_datetime(comp['datadate'])
comp['roa'] = comp['niq'] / comp['atq']
comp.to_parquet('compustat_fundq.parquet', index=False)
print(f"  Saved {len(comp):,} firm-quarter fundamentals")

db.close()
print("\n=== All supplement data pulled ===")
EOF