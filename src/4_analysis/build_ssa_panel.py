"""
Merge DeepSeek attribution codes back to firm-quarter panel.
Output: firm-quarter SSA scores ready for asset pricing tests.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load LLM codes
records = [json.loads(l) for l in open('production_deepseek.jsonl') 
           if 'error' not in json.loads(l)]
codes = pd.DataFrame(records)
print(f"LLM codes: {len(codes):,}")

# Load coding sample (has companyid, call_date, sue_terc_rank, permno, pair_id)
sample = pd.read_parquet('llm_coding_sample.parquet')
print(f"Coding sample: {len(sample):,}")

# Merge on pair_id
merged = sample.merge(codes, on='pair_id', how='inner')
print(f"Merged: {len(merged):,}")

# Compute response-level SSA components
internal_set = {'internal_action', 'internal_capability'}
external_set = {'external_structural', 'external_agentic'}

def cc(r):
    if r['outcome_valence']=='positive' and r['attribution_target'] in internal_set: return 1
    if r['outcome_valence']=='positive' and r['attribution_target'] in external_set: return -1
    return 0
def ext(r):
    if r['outcome_valence']=='negative' and r['attribution_target'] in external_set: return 1
    if r['outcome_valence']=='negative' and r['attribution_target'] in internal_set: return -1
    return 0

merged['credit_claiming'] = merged.apply(cc, axis=1)
merged['externalizing'] = merged.apply(ext, axis=1)
merged['ssa'] = merged['credit_claiming'] + merged['externalizing']

# Also: certainty-weighted SSA (robustness variant)
cert_w = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
merged['ssa_weighted'] = merged['ssa'] * merged['attribution_certainty'].map(cert_w)

# Aggregate to firm-quarter
fq = merged.groupby(['companyid', 'call_date']).agg(
    permno=('permno', 'first'),
    sue=('sue', 'first'),
    sue_terc_rank=('sue_terc_rank', 'first'),
    fpedats=('fpedats', 'first'),
    n_pairs=('pair_id', 'count'),
    ssa=('ssa', 'mean'),
    ssa_weighted=('ssa_weighted', 'mean'),
    credit_claiming=('credit_claiming', 'mean'),
    externalizing=('externalizing', 'mean'),
    pct_external=('attribution_target', lambda x: x.isin(external_set).mean()),
    pct_deflective=('responsiveness', lambda x: (x=='deflective').mean()),
    pct_low_certainty=('attribution_certainty', lambda x: (x=='low').mean()),
).reset_index()

print(f"\nFirm-quarter panel: {len(fq):,} events")
print(f"Unique firms: {fq['permno'].nunique():,}")
print(f"\nSSA by SUE tertile:")
print(fq.groupby('sue_terc_rank', observed=True)[['ssa','credit_claiming','externalizing']].mean())

fq.to_parquet('ssa_firm_quarter.parquet', index=False)
print(f"\nSaved to ssa_firm_quarter.parquet")