# Self-Serving Attribution in Earnings Calls

**An LLM-Based Measurement Framework and Within-Firm Evidence on Externalizing**

*Wenlan Xie · University of Chicago*

---

## Summary

This project develops an LLM-based framework for measuring **managerial self-serving
attribution (SSA)** in earnings-call discourse and provides within-firm evidence on
its predictive content. Using DeepSeek V3, we extract a structured attribution code
(outcome valence, attribution target, certainty, responsiveness, and a verbatim
supporting phrase) from **162,677 analyst-question / executive-answer pairs** drawn
from **137,245 S&P Capital IQ earnings-call transcripts** spanning 2019Q1–2023Q4.
Following the Bettman–Weitzel (1983) attribution framework, we decompose SSA into
two components: **credit-claiming** (internal attribution of positive outcomes) and
**externalizing** (external attribution of negative outcomes). The pipeline links
each call to IBES standardized unexpected earnings (SUE), CRSP returns, and Compustat
fundamentals, and conducts within-firm panel tests of whether attribution style
predicts subsequent earnings performance.

## Headline findings

- **Pervasive self-serving attribution.** Executives claim credit for positive
  outcomes in **78.9%** of relevant responses and externalize negative outcomes in
  **67.6%** of relevant responses. The asymmetry between internal attribution of
  good news and external attribution of bad news is highly significant
  (**χ² = 24,815**).

- **Externalizing predicts deteriorating fundamentals (within firm).** In two-way
  (firm + quarter) fixed-effects specifications, a one-standard-deviation increase in
  a firm's **externalizing** at a given call predicts a decline in that firm's
  **next-quarter SUE**:

  | Horizon | t-statistic | Interpretation |
  |---------|-------------|----------------|
  | h = +1 (next quarter) | **t = −3.88** | externalizing → lower future SUE |
  | h = +2 (two quarters) | **t = −3.01** | effect persists |
  | h = −1 (prior quarter, falsification) | **null** | no spurious reverse relation |

  The falsification test (predicting *past* SUE) returns a null, consistent with
  externalizing carrying forward-looking information rather than merely reflecting
  recent performance. The effect survives controls for current SUE, lagged SUE,
  credit-claiming, LLM-coded low-certainty (vague-language proxy), and deflective
  responsiveness.

## Data sources

All data are accessed through **WRDS** and are **license-restricted** (not
redistributable in this repository):

- **S&P Capital IQ Transcripts** (`ciq.*`) — earnings-call transcripts and components.
- **IBES** (`ibes.statsum_epsus`) — analyst EPS forecasts; SUE and forecast revisions.
- **CRSP** (`crsp.dsf`, Fama-French factors) — daily returns and risk factors.
- **Compustat** (`comp.fundq`) — quarterly fundamentals (ROA, earnings).
- **CRSP–IBES link** (`wrdsapps_link_crsp_ibes.ibcrsphist`) — ticker → PERMNO.

## Pipeline

```
                          ┌─────────────────────────────────────────────┐
                          │  WRDS  (CIQ Transcripts · IBES · CRSP · Comp) │
                          └─────────────────────────────────────────────┘
                                              │
  transcripts.py ─────────────────────────────┤  download raw transcripts + components
        │                                      │  → transcripts_data/*.parquet  (2.8 GB)
        ▼                                      │
  build_qa_pairs.py  (see note) ───────────────┤  pair analyst Q with executive A
        │                                      │  → qa_pairs.parquet
        ▼                                      │
  build_sue_link.py ───────────────────────────┤  CIQ id → CUSIP → IBES SUE → PERMNO
        │                                      │  → sue_panel.parquet
        ▼                                      │
  clean_sue.py ────────────────────────────────┤  filter + winsorize SUE
        │                                      │  → sue_panel_clean.parquet
        ▼                                      │
  finalize_sue.py ─────────────────────────────┤  within-quarter rank, keep top/bottom tertile
        │                                      │  → sue_panel_final.parquet
        ▼                                      │
  sample_qa_for_llm.py ────────────────────────┤  join Q-A pairs to analytic events, cap 5/event
        │                                      │  → llm_coding_sample.parquet
        ▼                                      │
  attribution_prompt.py  (prompt + schema, imported by the two runners below)
        │
        ├── run_llm_coding.py ────────────────┤  pilot: 500 pairs (synchronous)
        │                                      │  → pilot_deepseek.jsonl
        ▼                                      │
  run_production_deepseek.py ──────────────────┤  full 162,677 pairs (async, resumable)
        │                                      │  → production_deepseek.jsonl
        ▼                                      │
  build_ssa_panel.py ──────────────────────────┤  merge codes → firm-quarter SSA scores
        │                                      │  → ssa_firm_quarter.parquet
        ▼                                      │
  pull_crsp_ibes_supplement.py ────────────────┤  CRSP returns, FF factors, IBES revisions, Compustat
        │                                      │  → crsp_returns / ff_factors / ibes_revisions / compustat_fundq
        ▼                                      │
  asset_pricing_test.py ───────────────────────┤  portfolio sorts + Fama-MacBeth on SSA
        │                                      │  → ssa_with_returns.parquet
        ▼                                      │
  mechanism_test.py  /  mechanism_debug.py ─────┘  within-firm SUE/ROA/revision mechanism tests
```

> **Note on `build_qa_pairs.py`.** This script (which constructs `qa_pairs.parquet`
> by pairing each analyst question with the executive's answer from the raw
> transcript components) is a required pipeline step but is **not currently present
> in this repository**. It must be restored or re-implemented before the pipeline can
> be run end-to-end from scratch. Its inputs (`transcripts_data/`) and its output
> (`qa_pairs.parquet`) are both excluded from version control as restricted WRDS
> derivatives. See `docs/PIPELINE.md` for the expected schema.

## Environment setup

The project was developed against a conda environment named **`mlfin`** on Python 3.10.

```bash
# 1. Create and activate the environment
conda create -n mlfin python=3.10
conda activate mlfin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure WRDS credentials (creates ~/.pgpass on first connect)
python -c "import wrds; wrds.Connection()"

# 4. Set your DeepSeek API key (required by the LLM coding scripts)
export DEEPSEEK_API_KEY="sk-..."     # Windows: setx DEEPSEEK_API_KEY "sk-..."
```

### Required environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `DEEPSEEK_API_KEY` | `run_llm_coding.py`, `run_production_deepseek.py` | DeepSeek V4 API key (platform.deepseek.com). **Never hardcode.** |
| WRDS credentials | all `wrds`-using scripts | Stored in `~/.pgpass` after the first `wrds.Connection()`; prompted interactively otherwise. |

## Run instructions (raw download → final tables)

> **Run every script from the repository root** (not from inside `src/`), so the
> relative data paths in each script (e.g. `./llm_coding_sample.parquet`) resolve to
> the repo root where the pipeline reads and writes its datasets.

```bash
conda activate mlfin
export DEEPSEEK_API_KEY="sk-..."

# 1. Download raw CIQ transcripts (long; 2.8 GB output to transcripts_data/)
python src/1_ingest/transcripts.py

# 2. Build analyst-question / executive-answer pairs   (see note: build_qa_pairs.py)
python src/1_ingest/build_qa_pairs.py          # script not yet in repo — see note

# 3. Build the SUE linking panel (CIQ → CUSIP → IBES → PERMNO)
python src/2_link_sue/build_sue_link.py

# 4. Clean and winsorize SUE
python src/2_link_sue/clean_sue.py

# 5. Rank within quarter, keep top/bottom tertiles
python src/2_link_sue/finalize_sue.py

# 6. Sample Q-A pairs for LLM coding
python src/3_llm_coding/sample_qa_for_llm.py

# 7. (Optional) Pilot LLM run on 500 pairs
python src/3_llm_coding/run_llm_coding.py

# 8. Full production LLM coding (162,677 pairs; async + resumable)
python src/3_llm_coding/run_production_deepseek.py

# 9. Build firm-quarter SSA panel
python src/4_analysis/build_ssa_panel.py

# 10. Pull CRSP / FF / IBES / Compustat supplement
python src/4_analysis/pull_crsp_ibes_supplement.py

# 11. Asset-pricing tests (portfolio sorts + Fama-MacBeth)
python src/4_analysis/asset_pricing_test.py

# 12. Mechanism tests (within-firm next-quarter SUE, horizons, falsification)
python src/4_analysis/mechanism_test.py
python src/4_analysis/mechanism_debug.py
```

## Repository layout

```
earnings-call-attribution/
├── README.md  LICENSE  requirements.txt  .gitignore
├── data/
│   └── sample_production_output.jsonl     # first 100 LLM codes (illustrative)
├── docs/
│   ├── PIPELINE.md                        # detailed data-flow reference
│   └── paper/                             # LaTeX draft goes here
└── src/
    ├── 1_ingest/        transcripts.py    (+ build_qa_pairs.py — see note)
    ├── 2_link_sue/      build_sue_link.py · clean_sue.py · finalize_sue.py
    ├── 3_llm_coding/    attribution_prompt.py · sample_qa_for_llm.py
    │                    run_llm_coding.py · run_production_deepseek.py
    └── 4_analysis/      build_ssa_panel.py · pull_crsp_ibes_supplement.py
                         asset_pricing_test.py · mechanism_test.py · mechanism_debug.py
```

The large pipeline datasets (`*.parquet`, `transcripts_data/`, the full
`production_deepseek.jsonl`) are written to and read from the **repository root** and
are all git-ignored; only the illustrative sample under `data/` is committed.

## File map — scripts

Scripts live under `src/<stage>/` as shown in the layout above.

| Script | What it does |
|--------|--------------|
| `transcripts.py` | Downloads CIQ earnings-call transcripts and components from WRDS into 20 quarterly parquet files. |
| `build_qa_pairs.py` | *(missing — see note)* Pairs each analyst question with the executive's answer to produce `qa_pairs.parquet`. |
| `build_sue_link.py` | Links CIQ company IDs to CUSIP → IBES forecasts → CRSP PERMNO and computes per-call SUE. |
| `clean_sue.py` | Filters pathological denominators, applies analyst-count thresholds, and winsorizes SUE. |
| `finalize_sue.py` | Ranks SUE within calendar quarter and keeps the top and bottom tertiles as the analytic sample. |
| `sample_qa_for_llm.py` | Joins Q-A pairs to analytic events, applies substantive-length filters, caps at 5 pairs per call, and assigns stable `pair_id`s. |
| `attribution_prompt.py` | Defines the system/user prompts, JSON output schema, and the post-hoc SSA scoring function. |
| `run_llm_coding.py` | Pilot LLM coding on 500 pairs (synchronous DeepSeek calls); writes `pilot_deepseek.jsonl`. |
| `run_production_deepseek.py` | Full production LLM coding over all 162,677 pairs (async, concurrent, resumable). |
| `build_ssa_panel.py` | Merges LLM codes back to events, computes credit-claiming/externalizing/SSA, and aggregates to firm-quarter. |
| `pull_crsp_ibes_supplement.py` | Pulls CRSP daily returns, Fama-French factors, IBES revisions, and Compustat fundamentals. |
| `asset_pricing_test.py` | Quintile portfolio sorts and Fama-MacBeth regressions of forward returns on SSA. |
| `mechanism_test.py` | Tests whether SSA predicts next-quarter SUE, ROA, and analyst forecast revisions. |
| `mechanism_debug.py` | Within-firm panel + horizon (h=+1,+2,+3) and falsification (h=−1,−2) tests of externalizing on future SUE, with robustness controls. |

## File map — data outputs (regenerated locally; not in repo)

> All datasets below are **derivative works of paid WRDS data** and are excluded from
> version control. Re-run the pipeline to regenerate them.

| File | Contents |
|------|----------|
| `transcripts_data/transcripts_*.parquet` | Per-quarter event metadata: one row per earnings-call transcript. |
| `transcripts_data/components_*.parquet` | Per-quarter transcript components (speaker turns) with text. |
| `qa_pairs.parquet` | Analyst-question / executive-answer pairs with speaker and word-count fields. |
| `sue_panel.parquet` | One row per call with IBES SUE inputs and PERMNO link. |
| `sue_panel_clean.parquet` | Filtered, winsorized SUE panel. |
| `sue_panel_final.parquet` | Within-quarter-ranked analytic sample (top + bottom SUE tertiles). |
| `llm_coding_sample.parquet` | Q-A pairs selected for LLM coding (≤5 per event) with `pair_id`. |
| `pilot_deepseek.jsonl` | LLM attribution codes from the 500-pair pilot. |
| `production_deepseek.jsonl` | Full 162,677 LLM attribution codes (one JSON object per pair). |
| `ssa_firm_quarter.parquet` | Firm-quarter SSA scores (credit-claiming, externalizing, SSA, certainty/deflective shares). |
| `crsp_returns.parquet` | Daily CRSP returns around each call. |
| `ff_factors_daily.parquet` / `ff_factors_monthly.parquet` | Fama-French 5 factors + momentum. |
| `ibes_revisions.parquet` | IBES forecast snapshots for post-call revision measures. |
| `compustat_fundq.parquet` | Compustat quarterly fundamentals (NIQ, ATQ, ROA, etc.). |
| `ssa_with_returns.parquet` | Firm-quarter panel augmented with forward returns. |

### A note on reproducing `production_deepseek.jsonl`

The full 162,677-response LLM output is a derivative of restricted WRDS data and is
**not** included here. Only the first 100 records are provided, as
**`sample_production_output.jsonl`**, to illustrate the schema. The full file is fully
reproducible by re-running `run_production_deepseek.py` on `llm_coding_sample.parquet`
(itself regenerated by the pipeline). DeepSeek V3 is called at `temperature=0.0`, so
coding is approximately deterministic.

## Citation

```bibtex
@unpublished{xie2026ssa,
  title  = {Self-Serving Attribution in Earnings Calls: An LLM-Based
            Measurement Framework and Within-Firm Evidence on Externalizing},
  author = {Xie, Wenlan},
  year   = {2026},
  note   = {Working paper.}
}
```

## License

Source code is released under the [MIT License](LICENSE). The license does **not**
extend to any data derived from S&P Capital IQ, IBES, CRSP, or Compustat, which are
licensed through WRDS and may not be redistributed.
