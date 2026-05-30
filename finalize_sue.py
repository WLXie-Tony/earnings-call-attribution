"""
Quarter-cross-section rank SUE, sample bottom+top tertiles for LLM coding.
"""
import pandas as pd
from pathlib import Path

df = pd.read_parquet("./sue_panel_clean.parquet")
print(f"Input events: {len(df):,}")

# Calendar quarter
df['call_date'] = pd.to_datetime(df['call_date'])
df['call_qtr'] = df['call_date'].dt.to_period('Q')

# Within-quarter percentile rank (0 to 1)
df['sue_qrank'] = df.groupby('call_qtr')['sue'].rank(pct=True)

# Tertile based on rank
df['sue_terc_rank'] = pd.qcut(df['sue_qrank'], q=3, labels=['Bottom', 'Middle', 'Top'])

print("\n=== Within-quarter tertile composition ===")
print(df['sue_terc_rank'].value_counts())

# Sanity check: each tertile spread across quarters
print("\n=== Quarterly distribution (should be balanced) ===")
tab = df.groupby(['call_qtr', 'sue_terc_rank'], observed=True).size().unstack(fill_value=0)
print(tab.head(8))

# Final analytic sample: keep Bottom + Top
analytic = df[df['sue_terc_rank'].isin(['Bottom', 'Top'])].copy()
print(f"\n=== Analytic sample (Bottom + Top, full LLM coding target) ===")
print(f"Total events: {len(analytic):,}")
print(f"  Bottom: {(analytic['sue_terc_rank']=='Bottom').sum():,}")
print(f"  Top:    {(analytic['sue_terc_rank']=='Top').sum():,}")
print(f"Unique firms: {analytic['permno'].nunique():,}")
print(f"Date range: {analytic['call_date'].min().date()} to "
      f"{analytic['call_date'].max().date()}")

# Distribution check
print("\n=== SUE distribution by tertile ===")
print(analytic.groupby('sue_terc_rank', observed=True)['sue'].describe())

analytic.to_parquet("./sue_panel_final.parquet", index=False)
print("\nSaved to: sue_panel_final.parquet")