"""
Clean SUE panel: remove pathological denominators, winsorize tails.
"""
import pandas as pd
from pathlib import Path

IN_PATH = Path("./sue_panel.parquet")
OUT_PATH = Path("./sue_panel_clean.parquet")

df = pd.read_parquet(IN_PATH)
print(f"Raw events: {len(df):,}")

# Filter 1: minimum forecast dispersion + minimum analyst count
df = df[(df['stdev'] >= 0.01) & (df['numest'] >= 3)].copy()
print(f"After stdev≥0.01 & numest≥3: {len(df):,}")

# Recompute SUE (already done but just to be explicit)
df['sue_raw'] = (df['actual'] - df['medest']) / df['stdev']

# Winsorize at 1% / 99%
lo, hi = df['sue_raw'].quantile([0.01, 0.99])
df['sue'] = df['sue_raw'].clip(lower=lo, upper=hi)
print(f"\nWinsorization cutoffs: [{lo:.3f}, {hi:.3f}]")

print(f"\nCleaned SUE distribution:")
print(df['sue'].describe())

# Alternative SUE based on price-deflated forecast error (more standard in some papers)
# SUE2 = (actual - medest) / |medest|  — relative surprise
df['sue_pct'] = (df['actual'] - df['medest']) / df['medest'].abs()
df['sue_pct'] = df['sue_pct'].clip(*df['sue_pct'].quantile([0.01, 0.99]))

print(f"\nTertile cutoffs for SUE (standardized):")
print(f"  Bottom 33%: {df['sue'].quantile(0.33):.3f}")
print(f"  Top 33%:    {df['sue'].quantile(0.67):.3f}")

df['sue_tertile'] = pd.qcut(df['sue'], q=3, labels=['Bottom', 'Middle', 'Top'])
print(f"\nTertile composition:")
print(df['sue_tertile'].value_counts())

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
print(f"Final analytic events: {len(df):,}")