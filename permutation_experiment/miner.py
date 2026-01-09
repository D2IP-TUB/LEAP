#!/usr/bin/env python3
"""
Permutation miner: query a vLLM server for many random permutations of a single table.

Given:
  - a pandas DataFrame `df`
  - a question string `question`

This miner:
  - samples permutations of the row indices (without replacement)
  - uses vLLM's OpenAI-compatible API with chat messages and few-shot prompt
  - stores for each permutation:
        {"permutation": [...], "generation": "<model_output>"}
  - writes all results to a JSON file for offline reward computation later.
"""

import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import json
import random
from typing import Any, Dict, List

import pandas as pd
from openai import AsyncOpenAI

# --------------------
# CONFIG
# --------------------

VLLM_API_BASE = "http://localhost:8000/v1"  # vLLM OpenAI-compatible base URL
VLLM_MODEL_NAME = None  # None -> use first available model from /models, or set explicit name
NUM_PERMUTATIONS_TO_MINE = 6000
MAX_CONCURRENT_REQUESTS = 128  # tune based on your server
RANDOM_SEED = 42
DATA_FILE = "permutation_generations_miner.json"
MOCK_MODE = False  # True = fake responses, no server needed

random.seed(RANDOM_SEED)


# --------------------
# TABLE / PROMPT HELPERS
# --------------------

def df_to_header_rows(df: pd.DataFrame):
    """Split DataFrame into (header, rows) format."""
    header = list(map(str, df.columns))
    rows = [list(map(str, row)) for row in df.values]
    return header, rows


def table_to_csv(header: List[str], rows: List[List[Any]]) -> str:
    """Convert (header, rows) to CSV string matching your LEAP format."""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out)

    # Write header: leading space then column names
    writer.writerow([" "] + [str(col) for col in header])

    # Write rows as "row i,<cells...>"
    for i, row in enumerate(rows):
        writer.writerow([f"row {i}"] + [str(cell) for cell in row])

    return out.getvalue().strip()


def build_select_row_prompt(table_csv: str, question: str) -> List[Dict[str, str]]:
    """Build prompt for select_row action with few-shot examples."""
    system_message = """You are a helpful assistant that selects relevant rows from tables.
Given a table and a question, you should respond with select_row([row_indices]) containing the row numbers that are \
relevant to answering the question.
Only output the select_row function call, nothing else."""

    example_1_user = """Table:
 ,Home team,Home Team Score,Away Team,Away Team Score,Venue,Crowd
row 0,st kilda,13.12 (90),melbourne,13.11 (89),moorabbin oval,18836
row 1,south melbourne,9.12 (66),footscray,11.13 (79),lake oval,9154
row 2,richmond,20.17 (137),fitzroy,13.22 (100),mcg,27651

Question: Whose home team score is higher, richmond or st kilda?"""
    example_1_assistant = "select_row([0, 2])"

    example_2_user = """Table:
 ,home team,home team score,away team,away team score,venue,crowd
row 0,st kilda,13.12 (90),melbourne,13.11 (89),moorabbin oval,18836
row 1,south melbourne,9.12 (66),footscray,11.13 (79),lake oval,9154
row 2,richmond,20.17 (137),fitzroy,13.22 (100),mcg,27651
row 3,geelong,17.10 (112),collingwood,17.9 (111),kardinia park,23108
row 4,north melbourne,8.12 (60),carlton,23.11 (149),arden street oval,11271
row 5,hawthorn,15.16 (106),essendon,12.15 (87),vfl park,36749

Question: what is the away team with the highest score?"""
    example_2_assistant = "select_row([4])"

    current_user = f"""Table:
{table_csv}

Question: {question}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example_1_user},
        {"role": "assistant", "content": example_1_assistant},
        {"role": "user", "content": example_2_user},
        {"role": "assistant", "content": example_2_assistant},
        {"role": "user", "content": current_user},
    ]
    return messages


def apply_permutation(rows: List[List[Any]], permutation: List[int]) -> List[List[Any]]:
    """Apply permutation to table rows."""
    return [rows[i] for i in permutation]


# --------------------
# VLLM CALL
# --------------------

async def query_vllm(client: AsyncOpenAI, messages: List[Dict[str, str]]) -> str:
    """Send async request to vLLM server with chat messages; return raw text."""
    if MOCK_MODE:
        # Simple mock that always returns a fixed call
        await asyncio.sleep(0.01)
        return "select_row([0])"

    try:
        if VLLM_MODEL_NAME is None:
            # Query available models and use the first one
            models = await client.models.list()
            model_name = models.data[0].id if models.data else "default"
        else:
            model_name = VLLM_MODEL_NAME

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=64,
            temperature=0.0,
            n=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return ""


# --------------------
# MINER CORE
# --------------------

async def _mine_permutations_async(
    df: pd.DataFrame,
    question: str,
    num_permutations: int = NUM_PERMUTATIONS_TO_MINE,
) -> List[Dict[str, Any]]:
    """
    Core async miner: sample permutations without replacement, query vLLM for each,
    and return a list of dicts:
        {"permutation": [...], "generation": "<model_output>"}
    """
    header, rows = df_to_header_rows(df)
    n = len(rows)

    # For large n, full permutations are impossible; we just sample random permutations.
    seen = set()
    permutations: List[List[int]] = []

    while len(permutations) < num_permutations:
        p = list(range(n))
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            permutations.append(p)

    print(f"Mining {len(permutations)} permutations (n={n} rows)")

    client = AsyncOpenAI(
        base_url=VLLM_API_BASE,
        api_key="EMPTY",  # vLLM usually ignores API key
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_perm(p: List[int]) -> Dict[str, Any]:
        async with sem:
            permuted_rows = apply_permutation(rows, p)
            table_csv = table_to_csv(header, permuted_rows)
            messages = build_select_row_prompt(table_csv, question)
            text = await query_vllm(client, messages)
            return {
                "permutation": p,
                "generation": text,
            }

    # Fire off all requests at once; vLLM handles continuous batching underneath
    tasks = [process_perm(p) for p in permutations]
    results = []
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Mining permutations"):
        res = await coro
        results.append(res)

    return results


def run_miner(
    df: pd.DataFrame,
    question: str,
    num_permutations: int = NUM_PERMUTATIONS_TO_MINE,
    output_file: str = DATA_FILE,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper: mine permutations + generations and save to JSON.

    Usage from your training script:

        from miner_module import run_miner

        data = run_miner(df, item["question"], num_permutations=6000)
        # data is a list of {"permutation": [...], "generation": "..."}
    """
    results = asyncio.run(_mine_permutations_async(df, question, num_permutations))

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved {len(results)} permutation generations to {output_file}")
    return results


# Optional: quick CLI test
if __name__ == "__main__":
    import datasets

    ds = datasets.load_dataset("wikitablequestions", split="train")
    item = ds[0]
    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    question = item["question"]

    run_miner(df, question)
