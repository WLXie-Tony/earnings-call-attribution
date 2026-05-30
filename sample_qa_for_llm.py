"""
Sample Q-A pairs from analytic events (Bottom + Top SUE tertiles)
for LLM attribution coding.

Strategy:
  - Inner-join Q-A pairs with sue_panel_final on (companyid, call_date)
  - Filter substantive: q_word_count >= 15, a_word_count >= 50
  - Cap at 5 pairs per event, random sample if exceeds
  - Stratified by tertile for balance
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

PAIRS_PATH = Path("./qa_pairs.parquet")
SUE_PATH = Path("./sue_panel_final.parquet")
OUT_PATH = Path("./llm_coding_sample.parquet")

MAX_PAIRS_PER_EVENT = 5

print("Loading Q-A pairs...")
pairs = pd.read_parquet(PAIRS_PATH)
pairs['mostimportantdateutc'] = pd.to_datetime(pairs['mostimportantdateutc'])
pairs['companyid'] = pairs['companyid'].astype(int)
print(f"  Total: {len(pairs):,}")

# Substantive filter
pairs = pairs[(pairs['q_word_count'] >= 15) & (pairs['a_word_count'] >= 50)]
print(f"  Substantive (Q≥15w, A≥50w): {len(pairs):,}")

print("\nLoading SUE panel...")
sue = pd.read_parquet(SUE_PATH)
sue['call_date'] = pd.to_datetime(sue['call_date'])
sue['companyid'] = sue['companyid'].astype(int)
print(f"  Analytic events (Bottom+Top): {len(sue):,}")

# Inner join
print("\nMerging...")
merged = pairs.merge(
    sue[['companyid', 'call_date', 'sue', 'sue_terc_rank', 'permno',
         'fpedats', 'medest', 'actual', 'numest']],
    left_on=['companyid', 'mostimportantdateutc'],
    right_on=['companyid', 'call_date'],
    how='inner'
)
print(f"  Q-A pairs in analytic events: {len(merged):,}")
print(f"  Unique events covered: {merged.groupby(['companyid','call_date']).ngroups:,}")

# Sample max 5 pairs per event
print(f"\nCapping at {MAX_PAIRS_PER_EVENT} pairs per event (random)...")
def sample_group(g):
    if len(g) <= MAX_PAIRS_PER_EVENT:
        return g
    return g.sample(MAX_PAIRS_PER_EVENT, random_state=42)

sampled = merged.groupby(['companyid', 'call_date'], group_keys=False).apply(sample_group)
print(f"  Sampled pairs: {len(sampled):,}")

# Final stats
print("\n=== Final LLM coding sample ===")
print(f"Total Q-A pairs: {len(sampled):,}")
print(f"Unique events: {sampled.groupby(['companyid','call_date']).ngroups:,}")
print(f"Unique firms: {sampled['permno'].nunique():,}")
print(f"\nPairs per tertile:")
print(sampled['sue_terc_rank'].value_counts())
print(f"\nMean Q word count: {sampled['q_word_count'].mean():.0f}")
print(f"Mean A word count: {sampled['a_word_count'].mean():.0f}")
total_words = sampled['q_word_count'].sum() + sampled['a_word_count'].sum()
print(f"Total words (Q+A): {total_words:,}")
print(f"Estimated input tokens: {total_words*1.3/1e6:.1f}M")

# Assign a stable pair_id for tracking across LLM models
sampled = sampled.reset_index(drop=True)
sampled['pair_id'] = ['pair_' + str(i).zfill(7) for i in range(len(sampled))]

sampled.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")