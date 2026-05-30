"""
LLM attribution coding pipeline.
Pilot run: 500 random pairs with DeepSeek V3.
"""
import os
import pandas as pd
import json
import time
from pathlib import Path
from openai import OpenAI
from attribution_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, OUTPUT_SCHEMA, compute_ssa

# DeepSeek exposes an OpenAI-compatible API.
# Set the API key via environment variable (obtain from platform.deepseek.com):
#   export DEEPSEEK_API_KEY="sk-..."
_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not _api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY environment variable is not set. "
        "Export your key: export DEEPSEEK_API_KEY=sk-..."
    )
client = OpenAI(
    api_key=_api_key,
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"  # V3
PILOT_N = 500

# Load and sample
df = pd.read_parquet("./llm_coding_sample.parquet")
pilot = df.sample(PILOT_N, random_state=42).reset_index(drop=True)
print(f"Pilot sample: {len(pilot)} pairs")

results = []
out_path = Path(f"./pilot_deepseek.jsonl")

# Resume if interrupted
done_ids = set()
if out_path.exists():
    with open(out_path) as f:
        for line in f:
            done_ids.add(json.loads(line)['pair_id'])
    print(f"Resuming: {len(done_ids)} already done")

start = time.time()
for i, row in pilot.iterrows():
    if row['pair_id'] in done_ids:
        continue
    
    user_msg = USER_PROMPT_TEMPLATE.format(
        q_speaker=row['q_speaker'],
        q_text=row['q_text'][:2000],   # truncate extreme outliers
        a_speaker=row['a_speaker'],
        a_text=row['a_text'][:3000]
    )
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        parsed['pair_id'] = row['pair_id']
        parsed['model'] = MODEL
        parsed['input_tokens'] = resp.usage.prompt_tokens
        parsed['output_tokens'] = resp.usage.completion_tokens
        
        with open(out_path, 'a') as f:
            f.write(json.dumps(parsed) + '\n')
        results.append(parsed)
        
    except Exception as e:
        err = {'pair_id': row['pair_id'], 'error': str(e), 'model': MODEL}
        with open(out_path, 'a') as f:
            f.write(json.dumps(err) + '\n')
        print(f"  ERROR pair {row['pair_id']}: {e}")
    
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        eta = (len(pilot) - i - 1) / rate / 60
        print(f"  [{i+1}/{len(pilot)}] rate={rate:.1f}/s, ETA {eta:.1f} min")

print(f"\nDone. Saved to {out_path}")
print(f"Total time: {(time.time()-start)/60:.1f} min")