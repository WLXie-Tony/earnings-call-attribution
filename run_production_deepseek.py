"""
Production attribution coding with DeepSeek V3.
Async concurrent calls, resumable on crash, full 162k Q-A pair sample.
"""
import asyncio
import json
import time
import os
import signal
from pathlib import Path
import pandas as pd
from openai import AsyncOpenAI
from attribution_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# ============= CONFIG =============
# Set the DeepSeek API key via environment variable, e.g.:
#   export DEEPSEEK_API_KEY="sk-..."   (obtain from platform.deepseek.com)
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY environment variable is not set. "
        "Export your key: export DEEPSEEK_API_KEY=sk-..."
    )
MODEL = "deepseek-chat"
INPUT_PARQUET = "./llm_coding_sample.parquet"
OUTPUT_JSONL = Path("./production_deepseek.jsonl")
PROGRESS_LOG = Path("./production_progress.log")

CONCURRENCY = 8         # 8 个 async worker, DeepSeek 推荐 5-10
MAX_RETRIES = 3         # 单 call 失败重试次数
RETRY_BASE_DELAY = 2    # 指数 backoff base (秒)
SAVE_EVERY = 50         # 每 N 个 call flush 一次 disk
# =================================

client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
semaphore = asyncio.Semaphore(CONCURRENCY)

async def code_one_pair(row):
    """Single LLM call with retry-with-backoff."""
    user_msg = USER_PROMPT_TEMPLATE.format(
        q_speaker=row['q_speaker'],
        q_text=row['q_text'][:2000],
        a_speaker=row['a_speaker'],
        a_text=row['a_text'][:3000]
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=300,
                    timeout=60
                )
            parsed = json.loads(resp.choices[0].message.content)
            parsed['pair_id'] = row['pair_id']
            parsed['model'] = MODEL
            parsed['input_tokens'] = resp.usage.prompt_tokens
            parsed['output_tokens'] = resp.usage.completion_tokens
            return parsed
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {'pair_id': row['pair_id'], 'model': MODEL, 
                        'error': str(e)[:300]}
            await asyncio.sleep(RETRY_BASE_DELAY ** (attempt + 1))


async def main():
    # Load sample
    print(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Total pairs: {len(df):,}")
    
    # Resume: skip already-done pair_ids
    done = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['pair_id'])
                except Exception:
                    pass
        print(f"  Resuming: {len(done):,} already coded")
    
    todo = df[~df['pair_id'].isin(done)].to_dict('records')
    print(f"  To process: {len(todo):,}")
    if not todo:
        print("All done.")
        return
    
    # Counter for live progress
    total = len(todo)
    completed = 0
    errors = 0
    start_time = time.time()
    last_flush = time.time()
    
    # Buffer for batched disk writes
    buffer = []
    
    async def run_with_progress(row):
        nonlocal completed, errors
        result = await code_one_pair(row)
        if 'error' in result:
            errors += 1
        completed += 1
        return result
    
    # Schedule all tasks
    print(f"\nStarting {CONCURRENCY} concurrent workers...")
    tasks = [run_with_progress(row) for row in todo]
    
    # As tasks complete, write + log progress
    with open(OUTPUT_JSONL, 'a', buffering=1) as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            fout.write(json.dumps(result) + '\n')
            
            # Print progress every SAVE_EVERY
            if completed % SAVE_EVERY == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta_min = (total - completed) / rate / 60 if rate > 0 else 0
                msg = (f"[{time.strftime('%H:%M:%S')}] "
                       f"{completed}/{total} "
                       f"({100*completed/total:.1f}%) | "
                       f"rate={rate:.1f}/s | errors={errors} | "
                       f"ETA {eta_min:.0f} min")
                print(msg)
                with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
    
    total_time = (time.time() - start_time) / 60
    print(f"\n=== DONE ===")
    print(f"Total time: {total_time:.1f} min")
    print(f"Successful: {completed - errors}")
    print(f"Errors:     {errors}")


if __name__ == "__main__":
    asyncio.run(main())