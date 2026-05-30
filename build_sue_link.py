"""
Pull IBES SUE for all earnings calls in our sample, and build
CIQ companyid → IBES ticker → CRSP permno linking.

Strategy:
  (1) Get unique (companyid, earnings_call_date) from qa_pairs.parquet
  (2) Get CUSIP-9 for each companyid valid on call date (filter primaryflag=1)
  (3) Convert to CUSIP-8, query IBES statsum_epsus for forecast snapshots
      where statpers < call_date (last snapshot before call)
  (4) Compute SUE = (actual - medest) / stdev  using fpi='6' quarterly
  (5) Link to CRSP PERMNO via ibcrsphist with date filter

Output: sue_panel.parquet (one row per earnings call event with SUE + permno)
"""

import wrds
import pandas as pd
from pathlib import Path

QA_PATH = Path("./qa_pairs.parquet")
OUT_PATH = Path("./sue_panel.parquet")

db = wrds.Connection()

# Step 1: Get unique earnings call events from our sample
print("=" * 60)
print("Step 1: Loading earnings call universe")
print("=" * 60)
qa = pd.read_parquet(QA_PATH, columns=['companyid', 'mostimportantdateutc'])
events = qa.drop_duplicates(['companyid', 'mostimportantdateutc']).copy()
events['mostimportantdateutc'] = pd.to_datetime(events['mostimportantdateutc'])
events['companyid'] = events['companyid'].astype(int)
print(f"Unique earnings call events: {len(events):,}")
print(f"Unique companies: {events['companyid'].nunique():,}")
print(f"Date range: {events['mostimportantdateutc'].min()} to "
      f"{events['mostimportantdateutc'].max()}")

# Step 2: CIQ companyid → CUSIP (primary, time-valid)
print("\n" + "=" * 60)
print("Step 2: Pull CIQ → CUSIP mapping")
print("=" * 60)
companyids = events['companyid'].unique().tolist()
# Chunk into batches of 5000 to avoid huge IN clauses
cusip_chunks = []
for i in range(0, len(companyids), 5000):
    ids = companyids[i:i+5000]
    id_str = ','.join(str(x) for x in ids)
    q = f"""
    SELECT companyid, cusip, startdate, enddate, primaryflag
    FROM ciq.wrds_cusip
    WHERE companyid IN ({id_str})
      AND primaryflag = 1
    """
    cusip_chunks.append(db.raw_sql(q))
    print(f"  batch {i//5000 + 1}/{(len(companyids)+4999)//5000}: "
          f"{len(cusip_chunks[-1]):,} rows")

cusip_map = pd.concat(cusip_chunks, ignore_index=True)
cusip_map['companyid'] = cusip_map['companyid'].astype(int)
cusip_map['cusip_8'] = cusip_map['cusip'].str[:8]
cusip_map['startdate'] = pd.to_datetime(cusip_map['startdate'])
cusip_map['enddate'] = pd.to_datetime(cusip_map['enddate']).fillna(
    pd.Timestamp('2099-12-31'))
print(f"Total CUSIP mappings: {len(cusip_map):,}")

# Merge with date filter
events_cusip = events.merge(cusip_map, on='companyid', how='left')
mask = ((events_cusip['mostimportantdateutc'] >= events_cusip['startdate']) &
        (events_cusip['mostimportantdateutc'] <= events_cusip['enddate']))
events_cusip = events_cusip[mask].copy()
# Deduplicate: keep one cusip per event
events_cusip = events_cusip.drop_duplicates(['companyid', 'mostimportantdateutc'])
print(f"Events with valid CUSIP: {len(events_cusip):,} / {len(events):,} "
      f"({100*len(events_cusip)/len(events):.1f}%)")

# Step 3: Pull IBES statsum_epsus for our universe
print("\n" + "=" * 60)
print("Step 3: Pull IBES quarterly forecast snapshots")
print("=" * 60)
cusip8_list = events_cusip['cusip_8'].dropna().unique().tolist()
ibes_chunks = []
for i in range(0, len(cusip8_list), 2000):
    cs = cusip8_list[i:i+2000]
    cs_str = ','.join(f"'{x}'" for x in cs)
    q = f"""
    SELECT ticker, cusip, statpers, fpedats, fpi,
           medest, meanest, stdev, numest, actual, anndats_act
    FROM ibes.statsum_epsus
    WHERE cusip IN ({cs_str})
      AND fpi = '6'
      AND statpers >= '2018-10-01'
      AND statpers <= '2024-03-01'
      AND measure = 'EPS'
    """
    ibes_chunks.append(db.raw_sql(q))
    print(f"  batch {i//2000 + 1}/{(len(cusip8_list)+1999)//2000}: "
          f"{len(ibes_chunks[-1]):,} rows")

ibes = pd.concat(ibes_chunks, ignore_index=True)
ibes['statpers'] = pd.to_datetime(ibes['statpers'])
ibes['fpedats'] = pd.to_datetime(ibes['fpedats'])
ibes['anndats_act'] = pd.to_datetime(ibes['anndats_act'])
print(f"Total IBES forecast rows: {len(ibes):,}")

# Step 4: For each event, find latest forecast snapshot BEFORE earnings call
print("\n" + "=" * 60)
print("Step 4: Compute SUE per earnings call")
print("=" * 60)
events_cusip = events_cusip.rename(columns={'mostimportantdateutc': 'call_date'})

# Merge events with IBES on cusip, then filter to relevant forecast period
merged = events_cusip.merge(ibes, left_on='cusip_8', right_on='cusip', how='inner')
# Keep only forecasts (a) before the call, (b) for the fiscal period being reported
# Heuristic: fpedats should be within ±60 days of call_date (call reports prior quarter)
merged['days_to_fpe'] = (merged['call_date'] - merged['fpedats']).dt.days
merged = merged[(merged['days_to_fpe'] >= 0) & (merged['days_to_fpe'] <= 90)]
# Snapshot must be before the call (use forecast vintage)
merged = merged[merged['statpers'] < merged['call_date']]
# Take the LATEST snapshot per (event, fpedats)
merged = merged.sort_values(['companyid', 'call_date', 'fpedats', 'statpers'])
last_snap = merged.groupby(['companyid', 'call_date', 'fpedats']).tail(1)
# Per event, take the fpedats closest to call_date (most recent quarter)
last_snap = last_snap.sort_values(['companyid', 'call_date', 'days_to_fpe'])
final = last_snap.groupby(['companyid', 'call_date']).head(1).copy()

# Compute SUE
final['sue'] = (final['actual'] - final['medest']) / final['stdev']
# Drop pathological cases (stdev=0, missing actual)
final = final[final['stdev'] > 0]
final = final.dropna(subset=['sue', 'actual', 'medest'])
print(f"Events with computable SUE: {len(final):,}")

# Step 5: Link to CRSP PERMNO via ibcrsphist
print("\n" + "=" * 60)
print("Step 5: Link IBES ticker → CRSP PERMNO")
print("=" * 60)
link = db.raw_sql("""
    SELECT ticker, permno, ncusip, sdate, edate, score
    FROM wrdsapps_link_crsp_ibes.ibcrsphist
    WHERE score <= 6
""")
link['sdate'] = pd.to_datetime(link['sdate'])
link['edate'] = pd.to_datetime(link['edate'])
print(f"Linking table rows (score≤6): {len(link):,}")

final_link = final.merge(link, on='ticker', how='left')
mask = ((final_link['call_date'] >= final_link['sdate']) &
        (final_link['call_date'] <= final_link['edate']))
final_link = final_link[mask].copy()
# Keep best score per event
final_link = final_link.sort_values(['companyid', 'call_date', 'score'])
final_link = final_link.groupby(['companyid', 'call_date']).head(1)
print(f"Events with PERMNO: {len(final_link):,}")

# Output
keep_cols = ['companyid', 'call_date', 'fpedats', 'ticker', 'cusip_8',
             'permno', 'medest', 'meanest', 'stdev', 'numest', 'actual',
             'sue', 'anndats_act']
out = final_link[keep_cols].rename(columns={'cusip_8': 'cusip'})
out.to_parquet(OUT_PATH, index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Final analytic events: {len(out):,}")
print(f"Unique companies (PERMNOs): {out['permno'].nunique():,}")
print(f"\nSUE distribution:")
print(out['sue'].describe())
print(f"\nSUE tertile boundaries:")
print(f"  Bottom 33%: {out['sue'].quantile(0.33):.3f}")
print(f"  Top 33%: {out['sue'].quantile(0.67):.3f}")
print(f"\nSaved to: {OUT_PATH}")
db.close()