"""
WRDS Capital IQ Transcripts - Production Download
Paper: Managerial Self-Serving Attribution in Earnings Calls

Sample: All earnings calls 2019Q1-2023Q4
Output: 20 quarterly parquet files in ./transcripts_data/

Estimated runtime: 4-8 hours total (depends on WRDS server load)
Estimated total output: 5-10 GB
"""

import wrds
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

OUTPUT_DIR = Path("./transcripts_data")
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 2000  # transcripts per component-query batch

# Build 20-quarter schedule (2019Q1 to 2023Q4)
QUARTERS = []
for year in range(2019, 2024):
    for q, (sm, em) in enumerate([(1, 4), (4, 7), (7, 10), (10, 13)], 1):
        start = f"{year}-{sm:02d}-01"
        end = f"{year+1}-01-01" if em == 13 else f"{year}-{em:02d}-01"
        QUARTERS.append((f"{year}Q{q}", start, end))


def query_quarter(db, qname, start_date, end_date):
    # Step 1: latest transcript per earnings call event, length >= 10min
    transcript_q = f"""
    SELECT DISTINCT ON (keydevid)
        transcriptid, keydevid, companyid, companyname,
        mostimportantdateutc, mostimportanttimeutc,
        transcriptcollectiontypename, transcriptpresentationtypename,
        audiolengthsec, transcriptcreationdate_utc
    FROM ciq.wrds_transcript_detail
    WHERE keydeveventtypename = 'Earnings Calls'
      AND mostimportantdateutc >= '{start_date}'
      AND mostimportantdateutc < '{end_date}'
      AND audiolengthsec >= 600
    ORDER BY keydevid, transcriptcreationdate_utc DESC, transcriptid DESC
    """
    transcripts = db.raw_sql(transcript_q)
    print(f"    [Step 1] {len(transcripts):,} eligible transcripts")
    if len(transcripts) == 0:
        return None, None

    # Step 2: pull components in batches (avoid timeouts on huge IN clauses)
    tids = transcripts['transcriptid'].astype(int).tolist()
    n_batches = (len(tids) + BATCH_SIZE - 1) // BATCH_SIZE
    chunks = []
    for i in range(0, len(tids), BATCH_SIZE):
        batch_ids = tids[i:i + BATCH_SIZE]
        id_str = ','.join(str(x) for x in batch_ids)
        comp_q = f"""
        SELECT
            p.transcriptid, p.transcriptcomponentid, p.componentorder,
            p.speakertypeid, p.speakertypename,
            p.transcriptcomponenttypeid, p.transcriptcomponenttypename,
            p.transcriptpersonname, p.proid, p.companyofperson,
            p.word_count, c.componenttext
        FROM ciq.wrds_transcript_person p
        INNER JOIN ciq.ciqtranscriptcomponent c
            ON p.transcriptcomponentid = c.transcriptcomponentid
        WHERE p.transcriptid IN ({id_str})
        """
        chunk = db.raw_sql(comp_q)
        chunks.append(chunk)
        print(f"    [Step 2] Batch {i//BATCH_SIZE + 1}/{n_batches}: "
              f"{len(chunk):,} components")

    components = pd.concat(chunks, ignore_index=True)
    components = components.sort_values(
        ['transcriptid', 'componentorder']
    ).reset_index(drop=True)

    # Step 3: attach event-level metadata to each component
    meta_cols = ['transcriptid', 'companyid', 'companyname',
                 'mostimportantdateutc', 'keydevid', 'audiolengthsec']
    final = components.merge(transcripts[meta_cols], on='transcriptid', how='left')
    return transcripts, final


def main():
    db = wrds.Connection()
    log = open(OUTPUT_DIR / "download_log.txt", "a")
    log.write(f"\n=== Run started: {datetime.now()} ===\n")

    for qname, start, end in QUARTERS:
        comp_path = OUTPUT_DIR / f"components_{qname}.parquet"
        meta_path = OUTPUT_DIR / f"transcripts_{qname}.parquet"

        if comp_path.exists():
            print(f"\n[{qname}] already done — skip")
            continue

        print(f"\n[{qname}] {start} → {end}")
        t0 = time.time()
        try:
            transcripts, components = query_quarter(db, qname, start, end)
            if transcripts is None:
                print(f"  no data")
                continue
            transcripts.to_parquet(meta_path, index=False)
            components.to_parquet(comp_path, index=False, compression='snappy')
            mins = (time.time() - t0) / 60
            size_mb = comp_path.stat().st_size / 1e6
            msg = (f"  OK {len(transcripts):,} transcripts, "
                   f"{len(components):,} components, "
                   f"{size_mb:.0f} MB, {mins:.1f} min")
            print(msg)
            log.write(f"{qname}: {msg}\n")
            log.flush()
        except Exception as e:
            print(f"  ERROR: {e}")
            log.write(f"{qname}: ERROR — {e}\n")
            log.flush()
            try:
                db.close()
            except Exception:
                pass
            time.sleep(30)
            db = wrds.Connection()  # reconnect and continue

    log.write(f"=== Run ended: {datetime.now()} ===\n")
    log.close()
    db.close()


if __name__ == "__main__":
    main()