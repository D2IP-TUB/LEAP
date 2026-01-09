import asyncio
import json
import os
import re

import aiohttp
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment

# --- CONFIG ---
VLLM_URL = "http://localhost:8000/v1/chat/completions"
DATA_FILE = "permutation_generations_miner.json"  # mined by external miner script
TRAIN_EPOCHS = 100
BATCH_SIZE = 16  # currently unused, kept for future batching experiments


# =========================
# PROMPT BUILDING HELPERS (used only for evaluation)
# =========================

def df_to_table_csv(df: pd.DataFrame) -> str:
    """Render the DataFrame into the ' ,col1,col2,... / row i,...' format expected by the prompt."""
    csv_str = " ," + ",".join(map(str, df.columns)) + "\n"
    for i, row in enumerate(df.values):
        csv_str += f"row {i}," + ",".join([str(x) for x in row]) + "\n"
    return csv_str


def build_select_row_prompt(table_csv: str, question: str):
    """Build the prompt for the row selection task."""
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

    current_user = f"""Table:
{table_csv}

Question: {question}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example_1_user},
        {"role": "assistant", "content": example_1_assistant},
        {"role": "user", "content": current_user},
    ]
    return messages


# =========================
# PARSING & SCORING HELPERS
# =========================

def parse_select_row_indices(text):
    """
    Parse select_row([i, j, ...]) from the LLM output.

    Returns:
      - list[int]  -> parsed indices
      - []         -> parsed but empty selection
      - None       -> cannot be parsed at all (scenario 1)
    """
    if text is None:
        return None

    # More robust pattern: allow whitespace
    match = re.search(r"select\\?_row\(\[(.*?)\]\)", text)
    if not match:
        return None  # unparsable

    try:
        content = match.group(1)
        if not content.strip():
            return []  # parsed but no indices
        return [int(x.strip()) for x in content.split(",") if x.strip()]
    except Exception:
        return None  # treat parse failure as unparsable


def project_canonical_rows_to_permutation(permutation, canonical_rows):
    """
    Given:
      - permutation p: list where p[new_index] = original_index
      - canonical_rows: indices in the canonical (original) table

    Return:
      - list of indices (in permuted table) that correspond to those canonical rows.
    """
    p = list(permutation)
    pos = []
    for orig_idx in canonical_rows:
        if orig_idx in p:
            pos.append(p.index(orig_idx))
    return pos


def score_permutation_and_generation(permutation, text, canonical_correct_rows):
    """
    Implements your 3 scenarios:

    1. Generated text cannot be parsed -> scenario = 1 (bad)
    2. Generated action contains the right row(s) + other rows -> scenario = 2 (better but suboptimal)
    3. Generated action captures exactly the correct row(s) -> scenario = 3 (best)

    Also includes a "0" scenario for "parsed but wrong rows only" if you want to distinguish it.

    Returns a dict with:
      - permutation
      - generation (text)
      - selected_rows (parsed)
      - scenario (int)
      - reward (float)
    """
    if isinstance(canonical_correct_rows, int):
        canonical_correct_rows = [canonical_correct_rows]

    # 1) parse the prediction
    selected = parse_select_row_indices(text)

    # 2) map canonical correct rows to indices in the permuted table
    correct_in_perm = project_canonical_rows_to_permutation(permutation, canonical_correct_rows)
    correct_set = set(correct_in_perm)

    # 3) classify scenario & assign reward
    if selected is None:
        # Scenario 1: cannot be parsed at all
        scenario = 1
        reward = -1.0
    else:
        selected_set = set(selected)

        if not selected_set:
            # parsed but empty -> also pretty bad
            scenario = 1
            reward = -0.8
        elif correct_set.issubset(selected_set):
            # it contains all correct rows
            if selected_set == correct_set:
                # Scenario 3: exactly the correct row(s)
                scenario = 3
                reward = 1.0
            else:
                # Scenario 2: correct row(s) + extras
                scenario = 2
                # strictly less than 1, better than wrong; tunable
                reward = 0.6
        else:
            # parsed but does not include any correct rows
            scenario = 0
            reward = -0.5

    return {
        "permutation": permutation,
        "generation": text,
        "selected_rows": selected,  # may be None
        "scenario": scenario,
        "reward": reward,
    }


def compute_rewards(raw_results, canonical_correct_rows):
    scenario_counts = {
        "unparsable_or_empty": 0,       
        "wrong_rows_only": 0,           
        "correct_plus_extra_rows": 0,   
        "exact_correct_rows": 0         
    }

    enriched = []
    for entry in raw_results:
        p = entry["permutation"]
        text = entry["generation"]
        scored = score_permutation_and_generation(p, text, canonical_correct_rows)
        enriched.append(scored)

        # map numeric scenario → descriptive key
        if scored["scenario"] == 1:
            scenario_counts["unparsable_or_empty"] += 1
        elif scored["scenario"] == 0:
            scenario_counts["wrong_rows_only"] += 1
        elif scored["scenario"] == 2:
            scenario_counts["correct_plus_extra_rows"] += 1
        elif scored["scenario"] == 3:
            scenario_counts["exact_correct_rows"] += 1
    
    print("\nReward statistics (by scenario):")
    for k, v in scenario_counts.items():
        print(f"  {k}: {v}")

    return enriched


# =========================
# LLM CALL (for evaluation only)
# =========================

async def generate_answer(session, df, question):
    """Call the LLM and return the raw generation text (no scoring here)."""
    table_csv = df_to_table_csv(df)
    messages = build_select_row_prompt(table_csv, question)

    payload = {
        "messages": messages,
        "temperature": 0,
        # "model": "your-model-name",  # uncomment if your vLLM endpoint requires it
    }

    async with session.post(VLLM_URL, json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


# =========================
# TRAINING
# =========================

def train_offline(results_with_rewards, n):
    """
    Train the permutation model on the best-reward permutation.
    (Later we can generalize this to use more of the reward structure.)
    """
    best_entry = max(results_with_rewards, key=lambda x: x["reward"])
    target_idx = torch.tensor(best_entry["permutation"])
    print(f"\nTraining on Best Permutation (Reward: {best_entry['reward']}): {target_idx.tolist()}")

    log_alpha = nn.Parameter(torch.randn(n, n) * 0.1)
    optimizer = optim.Adam([log_alpha], lr=0.1)

    # Target Matrix Y: row i goes to position j
    Y = torch.zeros(n, n)
    Y[target_idx, torch.arange(n)] = 1.0

    print("\n--- Training log_alpha matrix (Offline) ---")
    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()

        # Sinkhorn iterations: approximate doubly-stochastic permutation matrix
        P = log_alpha.exp()
        for _ in range(10):
            P = P / P.sum(dim=-1, keepdim=True)
            P = P / P.sum(dim=-2, keepdim=True)

        loss = -torch.sum(Y * torch.log(P + 1e-9))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return log_alpha


# =========================
# UTIL: INFER CANONICAL CORRECT ROW(S)
# =========================

def infer_canonical_correct_rows(df: pd.DataFrame, answers):
    """
    Heuristic: find rows in the canonical table that contain any answer string.
    This gives you canonical_correct_rows if you don't want to hard-code them.
    """
    answers = [str(a).lower() for a in answers]
    candidate_rows = []
    for i, row in df.iterrows():
        row_text = " ".join(map(str, row.values)).lower()
        if any(ans in row_text for ans in answers):
            candidate_rows.append(i)
    return candidate_rows


# =========================
# MAIN
# =========================

def main():
    # 0. Load the WTQ instance (same one you mined on)
    ds = load_dataset("wikitablequestions", split="train")
    item = ds[0]

    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    question = item["question"]
    answers = item["answers"]

    # 1. Canonical correct rows (for the UNPERMUTED table)
    canonical_correct_rows = infer_canonical_correct_rows(df, answers)
    print(f"Inferred canonical correct rows: {canonical_correct_rows}")

    if not canonical_correct_rows:
        raise ValueError("Could not infer canonical correct rows; please set them manually.")

    # 2. Load mined permutation + generation data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file {DATA_FILE} not found. Run the miner first.")

    with open(DATA_FILE, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} mined permutations from {DATA_FILE}")

    # 3. OFFLINE REWARD COMPUTATION
    data_with_rewards = compute_rewards(raw_data, canonical_correct_rows)

    # 4. TRAIN PERMUTATION MODEL
    log_alpha = train_offline(data_with_rewards, len(df))

    # 5. EVALUATE & SAMPLE (fresh LLM calls, using learned permutations)
    print("\n--- FINAL EVALUATION: SAMPLING 8 PERMUTATIONS ---")

    async def evaluate_samples():
        async with aiohttp.ClientSession() as session:
            samples = []

            # Sample permutations from the learned matrix via Gumbel noise + Hungarian
            for _ in range(8):
                noise = -torch.empty_like(log_alpha).exponential_().log()  # Gumbel(0,1)
                perturbed_weights = log_alpha.detach() + noise
                _, perm = linear_sum_assignment(-perturbed_weights.numpy())
                samples.append(perm.tolist())

            print(f"Generated Samples: {samples}")

            results = []
            for i, p in enumerate(samples):
                text = await generate_answer(session, df.iloc[p], question)
                scored = score_permutation_and_generation(p, text, canonical_correct_rows)
                results.append(scored)
                print(
                    f"Sample {i + 1} | Permutation: {p} | "
                    f"Scenario: {scored['scenario']} | Reward: {scored['reward']} | "
                    f"Selected: {scored['selected_rows']}"
                )

            avg_reward = sum(r["reward"] for r in results) / len(results)
            print(f"\nAverage Reward of Model Samples: {avg_reward:.4f}")

    asyncio.run(evaluate_samples())


if __name__ == "__main__":
    main()
