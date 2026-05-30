# Pipeline reference

Detailed data flow for the self-serving attribution project. This complements the
high-level diagram in the root `README.md`. Every dataset described here is a
derivative of paid WRDS data and is excluded from version control; paths are relative
to the project root.

---

## Stage 0 — Raw transcript download

**Script:** `transcripts.py`
**Source:** `ciq.wrds_transcript_detail`, `ciq.wrds_transcript_person`,
`ciq.ciqtranscriptcomponent`
**Output:** `transcripts_data/transcripts_<YYYYQn>.parquet` (event metadata) and
`transcripts_data/components_<YYYYQn>.parquet` (speaker turns), for 2019Q1–2023Q4.

For each quarter the script selects the latest transcript per earnings-call event
(`keydeveventtypename = 'Earnings Calls'`, `audiolengthsec >= 600`), then pulls the
component-level text in batches of 2,000 transcript IDs to avoid query timeouts.
Components are attached to event-level metadata (company id, call date, keydevid,
audio length). The run is resumable: quarters whose `components_*.parquet` already
exists are skipped, and the WRDS connection is re-established on error.

Key fields per component: `transcriptid`, `componentorder`, `speakertypename`,
`transcriptcomponenttypename`, `transcriptpersonname`, `word_count`, `componenttext`,
plus merged `companyid`, `companyname`, `mostimportantdateutc`, `keydevid`,
`audiolengthsec`.

## Stage 1 — Question/answer pairing

**Script:** `build_qa_pairs.py` *(not present in this repository — must be restored)*
**Input:** `transcripts_data/components_*.parquet`
**Output:** `qa_pairs.parquet`

This step reconstructs the Q&A portion of each call by pairing an analyst question
turn with the executive's answer turn. Downstream scripts expect `qa_pairs.parquet`
to contain at least: `companyid`, `mostimportantdateutc`, `q_speaker`, `q_text`,
`q_word_count`, `a_speaker`, `a_text`, `a_word_count`. Because this script is missing,
it must be re-implemented to match that schema before the pipeline can run from raw
transcripts.

## Stage 2 — SUE construction and identifier linking

**Script:** `build_sue_link.py`
**Input:** `qa_pairs.parquet`
**Sources:** `ciq.wrds_cusip`, `ibes.statsum_epsus`,
`wrdsapps_link_crsp_ibes.ibcrsphist`
**Output:** `sue_panel.parquet`

1. Collect the unique `(companyid, call_date)` universe from the Q-A pairs.
2. Map each CIQ `companyid` to a primary, time-valid CUSIP (`primaryflag = 1`),
   filtering on the call date falling within `[startdate, enddate]`; reduce to CUSIP-8.
3. Pull quarterly IBES forecast snapshots (`fpi = '6'`, `measure = 'EPS'`).
4. For each event keep the latest forecast snapshot strictly **before** the call,
   for the fiscal period reported (`fpedats` within 0–90 days of the call).
5. Compute `SUE = (actual − medest) / stdev`; drop zero-dispersion / missing cases.
6. Link IBES ticker → CRSP `permno` via `ibcrsphist` (score ≤ 6), keeping the best
   score per event.

## Stage 3 — SUE cleaning

**Script:** `clean_sue.py`
**Input:** `sue_panel.parquet` → **Output:** `sue_panel_clean.parquet`

Keeps events with `stdev >= 0.01` and `numest >= 3`, recomputes SUE, and winsorizes
at the 1st/99th percentiles. Also computes an alternative percent-surprise measure
(`sue_pct`) and standardized tertiles for diagnostics.

## Stage 4 — Within-quarter ranking and analytic sample

**Script:** `finalize_sue.py`
**Input:** `sue_panel_clean.parquet` → **Output:** `sue_panel_final.parquet`

Ranks SUE within each calendar quarter (`sue_qrank`), forms within-quarter tertiles,
and keeps only the **Bottom** and **Top** tertiles as the analytic sample for LLM
coding (a balanced design across the SUE distribution).

## Stage 5 — Q-A sampling for LLM coding

**Script:** `sample_qa_for_llm.py`
**Inputs:** `qa_pairs.parquet`, `sue_panel_final.parquet`
**Output:** `llm_coding_sample.parquet`

Inner-joins Q-A pairs to analytic events on `(companyid, call_date)`, keeps
substantive exchanges (`q_word_count >= 15`, `a_word_count >= 50`), caps at 5 pairs
per event (random with `seed = 42`), and assigns a stable `pair_id`
(`pair_0000000`-style) so codes can be tracked across models.

## Stage 6 — Attribution prompt and schema

**Module:** `attribution_prompt.py` (imported, not run directly)

Defines the Bettman–Weitzel-based system prompt, the user prompt template, and the
JSON output schema with five required fields:

- `outcome_valence` ∈ {positive, negative, neutral, mixed}
- `attribution_target` ∈ {internal_action, internal_capability, external_structural,
  external_agentic, mixed_attribution, no_attribution}
- `attribution_certainty` ∈ {high, medium, low}
- `responsiveness` ∈ {direct, partial, deflective}
- `key_phrase` — a verbatim ≤15-word quote supporting the classification

It also provides `compute_ssa(row)`: +1 for internal attribution of positive outcomes,
−1 for external attribution of negative outcomes, 0 otherwise.

## Stage 7 — LLM coding

**Pilot:** `run_llm_coding.py` — 500 random pairs, synchronous, `pilot_deepseek.jsonl`.
**Production:** `run_production_deepseek.py` — all 162,677 pairs, async with a
concurrency-8 semaphore, retry-with-backoff, and resume-on-restart (skips `pair_id`s
already present in `production_deepseek.jsonl`). Both call DeepSeek V3
(`deepseek-chat`) at `temperature = 0.0`, `max_tokens = 300`, JSON response format.

**API key:** read from the `DEEPSEEK_API_KEY` environment variable (no hardcoded keys).

Each output line is a JSON object with the five coded fields plus `pair_id`, `model`,
`input_tokens`, `output_tokens` (or an `error` field on failure). See
`sample_production_output.jsonl` for the first 100 records.

## Stage 8 — Firm-quarter SSA panel

**Script:** `build_ssa_panel.py`
**Inputs:** `production_deepseek.jsonl`, `llm_coding_sample.parquet`
**Output:** `ssa_firm_quarter.parquet`

Merges LLM codes to the coding sample on `pair_id`, computes response-level
`credit_claiming` (±1) and `externalizing` (±1) and their sum `ssa`, plus a
certainty-weighted variant. Aggregates to firm-quarter, producing mean SSA components
along with `pct_external`, `pct_deflective`, and `pct_low_certainty` shares and event
metadata (`permno`, `sue`, `sue_terc_rank`, `fpedats`, `n_pairs`).

## Stage 9 — Market and fundamentals supplement

**Script:** `pull_crsp_ibes_supplement.py`
**Sources:** `crsp.dsf`, `ff.fivefactors_daily/_monthly`, `ibes.statsum_epsus`,
`comp.fundq`
**Outputs:** `crsp_returns.parquet`, `ff_factors_daily.parquet`,
`ff_factors_monthly.parquet`, `ibes_revisions.parquet`, `compustat_fundq.parquet`

Pulls daily returns for all sample PERMNOs, Fama-French 5 factors plus momentum, IBES
forecast snapshots for post-call revision measures, and Compustat quarterly
fundamentals (with `roa = niq / atq`).

## Stage 10 — Asset-pricing tests

**Script:** `asset_pricing_test.py`
**Inputs:** `ssa_firm_quarter.parquet`, `crsp_returns.parquet`,
`ff_factors_monthly.parquet`
**Output:** `ssa_with_returns.parquet`

Builds forward cumulative returns (1/3/6 months after the call), sorts firms into
within-quarter SSA quintiles, reports Q5−Q1 spreads, and runs Fama-MacBeth
cross-sectional regressions (Newey-West, 4 lags) of forward returns on standardized
SSA and on the credit-claiming / externalizing components separately and jointly.

## Stage 11 — Mechanism tests

**Scripts:** `mechanism_test.py`, `mechanism_debug.py`
**Input:** `ssa_firm_quarter.parquet` (plus `compustat_fundq`, `ibes_revisions`,
`sue_panel_final` for ROA/revision tests)

`mechanism_test.py` tests whether SSA predicts next-quarter SUE, ROA, and analyst
forecast revisions (Fama-MacBeth). `mechanism_debug.py` adds the main within-firm
panel evidence: two-way (firm + quarter) fixed-effects regressions of future SUE on
externalizing, horizon tests at h = +1, +2, +3, falsification tests at h = −1, −2,
and robustness controls for current/lagged SUE, credit-claiming, low-certainty, and
deflective responsiveness. The headline result — externalizing predicts lower
next-quarter SUE (t = −3.88 at h = +1, t = −3.01 at h = +2, null at h = −1) — comes
from this script.
