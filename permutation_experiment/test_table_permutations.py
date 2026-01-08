import re

import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment

# --- CONFIGURATION ---
USE_MOCK = False
VLLM_URL = "http://localhost:8000/v1/chat/completions"


# --- 1. YOUR PROMPT FUNCTION (PRESERVED) ---
def build_select_row_prompt(table_csv, question):
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


# --- 2. SINKHORN & DATA HELPERS ---
def stable_sinkhorn(log_alpha, n_iters=20, temp=0.1):
    P = log_alpha / temp
    for _ in range(n_iters):
        P = P - torch.logsumexp(P, dim=-1, keepdim=True)
        P = P - torch.logsumexp(P, dim=-2, keepdim=True)
    return torch.exp(P)


def gumbel_sinkhorn(log_alpha, n_iters=20, temp=0.1, hard=False):
    """
    Gumbel-Sinkhorn: Adds Gumbel noise for exploration, then applies Sinkhorn.
    If hard=True, uses straight-through estimator for discrete sampling with Hungarian matching.
    """
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_alpha) + 1e-20) + 1e-20)
    log_alpha_noisy = log_alpha + gumbel_noise

    # Apply Sinkhorn
    P_soft = stable_sinkhorn(log_alpha_noisy, n_iters=n_iters, temp=temp)

    if hard:
        # Use Hungarian algorithm to find optimal matching (no duplicates!)
        # Convert to numpy and solve assignment problem (maximize, so negate)
        cost_matrix = -P_soft.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Convert back to tensor
        indices = torch.from_numpy(col_ind).to(log_alpha.device)

        # Create hard permutation matrix
        P_hard = torch.zeros_like(P_soft)
        P_hard[row_ind, col_ind] = 1.0

        # Straight-through estimator: hard selection in forward, soft in backward
        P = P_hard - P_soft.detach() + P_soft  # Gradient flows through P_soft
        return P, indices
    else:
        return P_soft, None


def df_to_indexed_csv(df):
    csv_str = " ," + ",".join(df.columns) + "\n"
    for i, row in enumerate(df.values):
        csv_str += f"row {i}," + ",".join([str(x) for x in row]) + "\n"
    return csv_str


# --- 3. THE LLM SCORER WITH PARTIAL REWARDS ---
def get_llm_reward(df, question, ground_truth):
    table_csv = df_to_indexed_csv(df)
    messages = build_select_row_prompt(table_csv, question)

    if USE_MOCK:
        generated_text = "select_row([0, 1])"
    else:
        payload = {"messages": messages, "temperature": 0}
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=10)
            generated_text = r.json()["choices"][0]["message"]["content"]
            print(f"LLM Response: {generated_text}")
        except Exception as e:
            print(f"vLLM Error: {e}")
            return 0.0

    match = re.search(r"select\\?_row\(\[(.*?)\]\)", generated_text)
    if not match:
        return 0.0

    try:
        selected_indices = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        print(f"resulted selection ids after processing {selected_indices}")
        if not selected_indices:
            return 0.0

        has_answer = False
        num_correct_rows = 0

        for idx in selected_indices:
            row_text = " ".join(df.iloc[[idx]].astype(str).values.flatten()).lower()
            print(f"context selected: {row_text}")
            row_hit = any(str(ans).lower() in row_text for ans in ground_truth)
            if row_hit:
                has_answer = True
                num_correct_rows += 1

        if not has_answer:
            return -0.1

        precision = num_correct_rows / len(selected_indices)
        return 0.5 + (0.5 * precision)

    except Exception as e:
        print(f"Scoring Error: {e}")
        return 0.0


# --- 4. MAIN OPTIMIZATION LOOP WITH EXPLORATION ---
def run_experiment(dataset_index=0):
    ds = load_dataset("wikitablequestions", split="train")
    item = ds[dataset_index]
    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    question = item["question"]
    answers = item["answers"]

    print(f"initial question:\n {question}")
    print(f"initial table:\n {df}")

    n = len(df)
    log_alpha = nn.Parameter(torch.randn(n, n) * 0.1)
    optimizer = optim.Adam([log_alpha], lr=0.05)

    print(f"--- Starting Optimization for Table {dataset_index} ---")
    print(f"Question: {question}")

    # Temperature schedule: start high for exploration, anneal down
    def temp_schedule(epoch):
        return max(1.0 * (0.95 ** (epoch / 10)), 0.1)

    # Entropy bonus schedule: higher early for exploration
    def entropy_weight_schedule(epoch):
        return max(0.01 * (0.99**epoch), 0.001)

    best_reward = -float("inf")
    best_indices = None

    for epoch in range(51):
        optimizer.zero_grad()

        # Get current temperature
        temp = temp_schedule(epoch)
        entropy_weight = entropy_weight_schedule(epoch)

        # Sample permutation with Gumbel-Sinkhorn (exploration!)
        P_soft, indices = gumbel_sinkhorn(log_alpha, n_iters=20, temp=temp, hard=True)

        # Evaluate this permutation
        permuted_df = df.iloc[indices.tolist()]
        reward = get_llm_reward(permuted_df, question, answers)

        # Track best permutation
        if reward > best_reward:
            best_reward = reward
            best_indices = indices.clone()

        # Compute log probabilities from the soft distribution
        log_probs = torch.log(P_soft + 1e-9)
        picked_log_probs = log_probs[torch.arange(n), indices]

        # Policy gradient loss with baseline (use best reward as baseline)
        baseline = best_reward if epoch > 0 else 0.0
        advantage = reward - baseline
        policy_loss = -advantage * picked_log_probs.sum()

        # Entropy bonus for exploration (encourage diversity)
        entropy = -(P_soft * torch.log(P_soft + 1e-9)).sum()
        entropy_bonus = -entropy_weight * entropy

        # Total loss
        loss = policy_loss + entropy_bonus

        # Always compute gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_alpha], max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Reward: {reward:+.4f} | Best: {best_reward:+.4f} | "
                f"Temp: {temp:.3f} | Entropy: {entropy.item():.3f} | Loss: {loss.item():.4f}"
            )

    # Use best permutation for final result
    with torch.no_grad():
        final_df = df.iloc[best_indices.tolist()] if best_indices is not None else df

        print("\n--- Final Permuted Table (Best) ---")
        print(final_df.to_string(index=False))

        print(f"generated answer is: {get_llm_reward(final_df, question, answers)}")
        print(f"correct answers are: {answers}")
        print(f"Best reward achieved: {best_reward:.4f}")


if __name__ == "__main__":
    run_experiment(2)
