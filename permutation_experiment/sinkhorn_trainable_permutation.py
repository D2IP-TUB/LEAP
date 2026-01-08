import asyncio
import re

import aiohttp
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
USE_MOCK = False
VLLM_URL = "http://localhost:8000/v1/chat/completions"
BATCH_SIZE = 12  # Number of permutations to sample and evaluate in parallel per epoch
USE_GENERALIZABLE_MODEL = True  # Set to True to use neural model, False for per-table optimization
EPOCHS_PER_TABLE = 30  # Reduced from 51 for faster multi-table training


# --- 1. GENERALIZABLE PERMUTATION MODEL ---
class PermutationModel(nn.Module):
    """
    Neural model that generates permutation logits for any table size.

    Approach: Score how well each (row_content, question, position) triple fits together.
    """

    def __init__(self, embedding_dim=384, hidden_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = embedding_dim

        # Position embedding: learnable embeddings for positions 0-99
        self.max_positions = 100
        self.position_embedding = nn.Embedding(self.max_positions, 64)

        # Neural network to score (row_embedding, question_embedding, position) → score
        input_dim = embedding_dim * 2 + 64  # row + question + position
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def embed_table_and_question(self, df, question):
        """Embed rows and question using sentence transformer."""
        # Embed each row
        row_texts = [" ".join(map(str, row)) for row in df.values]
        row_embeddings = self.embedding_model.encode(row_texts, convert_to_tensor=True)

        # Embed question
        question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)[0]

        # Move embeddings to model's device
        row_embeddings = row_embeddings.to(self.device)
        question_embedding = question_embedding.to(self.device)

        return row_embeddings, question_embedding

    def forward(self, df, question, cached_embeddings=None):
        """
        Generate n×n matrix of logits for permutation.

        logits[i, j] = score for "row i should go in position j"

        Args:
            df: DataFrame
            question: Question string
            cached_embeddings: Optional tuple of (row_embs, q_emb) to avoid recomputing
        """
        n = len(df)

        # Get embeddings (use cache if available)
        if cached_embeddings is not None:
            row_embs, q_emb = cached_embeddings
        else:
            row_embs, q_emb = self.embed_table_and_question(df, question)

        # Create position embeddings (0, 1, 2, ..., n-1)
        positions = torch.arange(n, device=row_embs.device)
        pos_embs = self.position_embedding(positions)  # (n, 64)

        # OPTIMIZATION: Vectorized computation instead of loop
        # Create all (row, position) pairs at once
        # row_embs: (n, emb_dim) -> (n, 1, emb_dim) -> (n, n, emb_dim)
        row_repeated = row_embs.unsqueeze(1).expand(n, n, -1)  # (n, n, emb_dim)

        # q_emb: (emb_dim,) -> (1, 1, emb_dim) -> (n, n, emb_dim)
        q_repeated = q_emb.unsqueeze(0).unsqueeze(0).expand(n, n, -1)  # (n, n, emb_dim)

        # pos_embs: (n, 64) -> (1, n, 64) -> (n, n, 64)
        pos_repeated = pos_embs.unsqueeze(0).expand(n, -1, -1)  # (n, n, 64)

        # Concatenate all features: (n, n, emb_dim*2 + 64)
        combined = torch.cat([row_repeated, q_repeated, pos_repeated], dim=2)  # (n, n, input_dim)

        # Score all (row, position) pairs at once
        # Reshape to (n*n, input_dim), pass through scorer, reshape back
        combined_flat = combined.view(n * n, -1)  # (n*n, input_dim)
        scores_flat = self.scorer(combined_flat).squeeze(-1)  # (n*n,)
        logits = scores_flat.view(n, n)  # (n, n)

        return logits  # (n, n)

    def to(self, device):
        """Override to() to also update self.device attribute."""
        self.device = device
        return super().to(device)


# --- 2. YOUR PROMPT FUNCTION (PRESERVED) ---
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
def parse_and_score_response(generated_text, df, ground_truth):
    """
    Helper function to parse LLM response and compute reward.

    Reward structure:
    - Not parsable / Error: -1.0 (worst - LLM didn't follow format)
    - Empty selection: -0.8 (very bad - LLM gave up)
    - No correct rows (all false positives): -0.5 (bad - wrong selection)
    - Mixed (some correct, some wrong): 0.0 to 0.8 based on F1 score
    - Perfect (only correct rows, no false positives): 1.0 (best)
    """
    match = re.search(r"select\\?_row\(\[(.*?)\]\)", generated_text)
    if not match:
        # print("Failed to parse LLM response - format error")
        return -1.0  # Larger penalty for not following format

    try:
        selected_indices = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        # print(f"resulted selection ids after processing {selected_indices}")

        if not selected_indices:
            # print("Empty selection")
            return -0.8  # Large penalty for empty selection

        # Count true positives and false positives
        num_correct_rows = 0  # true positives
        num_wrong_rows = 0  # false positives

        for idx in selected_indices:
            row_text = " ".join(df.iloc[[idx]].astype(str).values.flatten()).lower()
            # print(f"context selected: {row_text}")
            row_hit = any(str(ans).lower() in row_text for ans in ground_truth)
            if row_hit:
                num_correct_rows += 1
            else:
                num_wrong_rows += 1

        # If no correct rows at all (all false positives)
        if num_correct_rows == 0:
            # print("No correct rows selected")
            return -0.5  # Moderate penalty for selecting wrong rows

        # Calculate precision and recall-like metrics
        precision = num_correct_rows / len(selected_indices)

        # Perfect score only if no false positives
        if num_wrong_rows == 0:
            # print(f"Perfect selection! {num_correct_rows} correct rows, 0 false positives")
            return 1.0

        # For mixed results, use F1-like score scaled to [0, 0.8]
        # This penalizes both false positives and gives partial credit
        # F1 = 2 * precision (we don't have true recall measure)
        # Scale to [0, 0.8] so perfect (1.0) is reserved for no false positives
        f1_score = 2 * precision - 1  # Maps precision [0.5, 1.0] to [0, 1.0]
        reward = max(0.0, min(0.8, f1_score * 0.8))

        # print(f"Mixed result: {num_correct_rows} correct, {num_wrong_rows} wrong, precision={precision:.2f}, reward={reward:.2f}")
        return reward

    except Exception as e:
        print(f"Scoring Error: {e}")
        return -1.0  # Larger penalty for errors


async def get_llm_reward_async(session, df, question, ground_truth, sample_idx):
    """Async function to get reward for a single permuted table."""
    table_csv = df_to_indexed_csv(df)
    messages = build_select_row_prompt(table_csv, question)

    if USE_MOCK:
        generated_text = "select_row([0, 1])"
    else:
        payload = {"messages": messages, "temperature": 0}
        try:
            async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                result = await response.json()
                generated_text = result["choices"][0]["message"]["content"]
                # print(f"Sample {sample_idx} LLM Response: {generated_text}")
        except Exception as e:
            print(f"Sample {sample_idx} vLLM Error: {e}")
            return 0.0

    return parse_and_score_response(generated_text, df, ground_truth)


async def get_rewards_batch(dfs, question, ground_truth):
    """Get rewards for multiple permuted tables concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [get_llm_reward_async(session, df, question, ground_truth, idx) for idx, df in enumerate(dfs)]
        return await asyncio.gather(*tasks)


def get_llm_reward(df, question, ground_truth):
    """Synchronous wrapper for backward compatibility."""
    return asyncio.run(get_rewards_batch([df], question, ground_truth))[0]


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

    # Initialize model or per-table parameters
    if USE_GENERALIZABLE_MODEL:
        print("Using generalizable PermutationModel")
        model = PermutationModel(embedding_dim=384, hidden_dim=256)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower LR for neural model
        best_model_state = None
    else:
        print("Using per-table optimization")
        # Better initialization: start close to identity permutation
        log_alpha = nn.Parameter(torch.eye(n) * 2.0 + torch.randn(n, n) * 0.01)
        optimizer = optim.Adam([log_alpha], lr=0.01)  # Lower learning rate for stability
        best_log_alpha = log_alpha.clone().detach()

    print(f"--- Starting Optimization for Table {dataset_index} ---")
    print(f"Question: {question}")

    # Temperature schedule: start high for exploration, anneal down
    def temp_schedule(epoch):
        return max(1.0 * (0.95 ** (epoch / 10)), 0.1)

    # Entropy bonus schedule: higher early for exploration
    def entropy_weight_schedule(epoch):
        return max(0.01 * (0.99**epoch), 0.001)

    best_reward = -float("inf")

    # Moving average baseline for stability
    baseline_ema = 0.0
    baseline_ema_alpha = 0.1

    # Track training progress
    training_history = {
        "epoch": [],
        "avg_reward": [],
        "best_reward": [],
        "min_reward": [],
        "max_reward": [],
    }

    # OPTIMIZATION: Pre-compute embeddings for generalizable model
    if USE_GENERALIZABLE_MODEL:
        with torch.no_grad():
            cached_embeddings = model.embed_table_and_question(df, question)
    else:
        cached_embeddings = None

    for epoch in range(51):
        optimizer.zero_grad()

        # Get current temperature
        temp = temp_schedule(epoch)
        entropy_weight = entropy_weight_schedule(epoch)

        # Sample multiple permutations with Gumbel-Sinkhorn (exploration!)
        batch_P_soft = []
        batch_indices = []
        batch_permuted_dfs = []

        for _ in range(BATCH_SIZE):
            # Get log_alpha from model or use per-table parameter
            if USE_GENERALIZABLE_MODEL:
                log_alpha = model(df, question, cached_embeddings=cached_embeddings)

            P_soft, indices = gumbel_sinkhorn(log_alpha, n_iters=20, temp=temp, hard=True)
            batch_P_soft.append(P_soft)
            batch_indices.append(indices)
            batch_permuted_dfs.append(df.iloc[indices.tolist()])

        # Evaluate all permutations in parallel
        rewards = asyncio.run(get_rewards_batch(batch_permuted_dfs, question, answers))

        # Update moving average baseline
        current_batch_avg = sum(rewards) / len(rewards)
        if epoch == 0:
            baseline_ema = current_batch_avg
        else:
            baseline_ema = (1 - baseline_ema_alpha) * baseline_ema + baseline_ema_alpha * current_batch_avg

        # Track best permutation and save best model
        for i, reward in enumerate(rewards):
            if reward > best_reward:
                best_reward = reward
                if USE_GENERALIZABLE_MODEL:
                    import copy

                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    best_log_alpha = log_alpha.clone().detach()

        # Compute policy gradient loss for all samples in batch
        total_policy_loss = 0.0
        total_entropy = 0.0

        for i in range(BATCH_SIZE):
            P_soft = batch_P_soft[i]
            indices = batch_indices[i]
            reward = rewards[i]

            # Compute log probabilities from the soft distribution
            log_probs = torch.log(P_soft + 1e-9)
            picked_log_probs = log_probs[torch.arange(n), indices]

            # Policy gradient loss with EMA baseline (more stable)
            advantage = reward - baseline_ema
            total_policy_loss += -advantage * picked_log_probs.sum()

            # Entropy bonus for exploration (encourage diversity)
            entropy = -(P_soft * torch.log(P_soft + 1e-9)).sum()
            total_entropy += entropy

        # Average across batch
        policy_loss = total_policy_loss / BATCH_SIZE
        avg_entropy = total_entropy / BATCH_SIZE
        entropy_bonus = -entropy_weight * avg_entropy

        # Total loss
        loss = policy_loss + entropy_bonus

        # Only update if loss is reasonable (prevent exploding gradients)
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            if USE_GENERALIZABLE_MODEL:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_([log_alpha], max_norm=0.5)  # Stricter clipping
            optimizer.step()
        else:
            print(f"Warning: Skipping update due to invalid loss: {loss.item()}")

        # Record training metrics
        avg_reward = sum(rewards) / len(rewards)
        training_history["epoch"].append(epoch)
        training_history["avg_reward"].append(avg_reward)
        training_history["best_reward"].append(best_reward)
        training_history["min_reward"].append(min(rewards))
        training_history["max_reward"].append(max(rewards))

        if epoch % 10 == 0:
            loss_val = loss.item() if not torch.isnan(loss) else float("nan")
            print(
                f"Epoch {epoch:03d} | Avg Reward: {avg_reward:+.4f} | Best: {best_reward:+.4f} | "
                f"Baseline: {baseline_ema:+.4f} | Temp: {temp:.3f} | Entropy: {avg_entropy.item():.3f} | Loss: {loss_val:.4f}"
            )

    # Print training improvement summary
    print("\n" + "=" * 80)
    print("TRAINING IMPROVEMENT SUMMARY")
    print("=" * 80)

    initial_avg = training_history["avg_reward"][0]
    final_avg = training_history["avg_reward"][-1]
    initial_best = training_history["best_reward"][0]
    final_best = training_history["best_reward"][-1]

    print("\nInitial Performance (Epoch 0):")
    print(f"  Average Reward: {initial_avg:+.4f}")
    print(f"  Best Reward:    {initial_best:+.4f}")

    print(f"\nFinal Performance (Epoch {len(training_history['epoch']) - 1}):")
    print(f"  Average Reward: {final_avg:+.4f}")
    print(f"  Best Reward:    {final_best:+.4f}")

    print("\nImprovement:")
    print(f"  Average Reward: {final_avg - initial_avg:+.4f} ({100 * (final_avg - initial_avg) / (abs(initial_avg) + 1e-6):+.1f}%)")
    print(f"  Best Reward:    {final_best - initial_best:+.4f} ({100 * (final_best - initial_best) / (abs(initial_best) + 1e-6):+.1f}%)")

    # Show reward progression every 10 epochs
    print("\nReward Progression:")
    print(f"{'Epoch':<8} {'Avg Reward':<12} {'Best Reward':<12} {'Min Reward':<12} {'Max Reward':<12}")
    print("-" * 60)
    for i in range(0, len(training_history["epoch"]), 10):
        epoch = training_history["epoch"][i]
        avg_r = training_history["avg_reward"][i]
        best_r = training_history["best_reward"][i]
        min_r = training_history["min_reward"][i]
        max_r = training_history["max_reward"][i]
        print(f"{epoch:<8} {avg_r:<+12.4f} {best_r:<+12.4f} {min_r:<+12.4f} {max_r:<+12.4f}")

    # Show final epoch if not already shown
    if (len(training_history["epoch"]) - 1) % 10 != 0:
        epoch = training_history["epoch"][-1]
        avg_r = training_history["avg_reward"][-1]
        best_r = training_history["best_reward"][-1]
        min_r = training_history["min_reward"][-1]
        max_r = training_history["max_reward"][-1]
        print(f"{epoch:<8} {avg_r:<+12.4f} {best_r:<+12.4f} {min_r:<+12.4f} {max_r:<+12.4f}")

    print("\n" + "=" * 80)

    # Evaluate final results
    with torch.no_grad():
        print("\n" + "=" * 80)
        print("FINAL EVALUATION - LEARNED POLICY")
        print("=" * 80)

        # Evaluate using the best saved model (more reliable)
        print("\n--- Using Best Model from Training ---")

        if USE_GENERALIZABLE_MODEL:
            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            log_alpha_best = model(df, question)
            P_best, indices_best = gumbel_sinkhorn(log_alpha_best, n_iters=20, temp=0.01, hard=True)
        else:
            P_best, indices_best = gumbel_sinkhorn(best_log_alpha, n_iters=20, temp=0.01, hard=True)

        best_model_df = df.iloc[indices_best.tolist()]

        print(best_model_df.to_string(index=False))
        print(f"\nPermutation indices: {indices_best.tolist()}")

        best_model_reward = get_llm_reward(best_model_df, question, answers)
        print(f"\nBest Model Reward: {best_model_reward:.4f}")

        # Also evaluate current final model
        print("\n--- Current Final Model ---")

        if USE_GENERALIZABLE_MODEL:
            # Don't need to load state, model is already at final state
            log_alpha_current = model(df, question)
            P_current, indices_current = gumbel_sinkhorn(log_alpha_current, n_iters=20, temp=0.01, hard=True)
        else:
            P_current, indices_current = gumbel_sinkhorn(log_alpha, n_iters=20, temp=0.01, hard=True)

        current_model_df = df.iloc[indices_current.tolist()]

        print(current_model_df.to_string(index=False))
        print(f"\nPermutation indices: {indices_current.tolist()}")

        current_model_reward = get_llm_reward(current_model_df, question, answers)
        print(f"\nCurrent Model Reward: {current_model_reward:.4f}")

        # Comparison
        print("\n--- Summary ---")
        print(f"Best Model Reward:     {best_model_reward:.4f}")
        print(f"Current Model Reward:  {current_model_reward:.4f}")
        print(f"Best Training Reward:  {best_reward:.4f}")
        print(f"Correct Answers:       {answers}")

        if best_model_reward >= best_reward * 0.95:  # Within 5% is good
            print("\n✓ Model successfully learned good permutation!")
        else:
            print("\n⚠ Model may need more training or different hyperparameters")

    # Save the model if using generalizable model
    if USE_GENERALIZABLE_MODEL and best_model_state is not None:
        model_save_path = f"permutation_model_table_{dataset_index}.pt"
        torch.save(best_model_state, model_save_path)
        print(f"\n💾 Saved best model to {model_save_path}")

    return best_reward


def train_on_multiple_tables(dataset_indices, epochs_per_table=51):
    """
    Train the generalizable model on multiple tables.
    This allows the model to learn patterns across different tables.
    """
    if not USE_GENERALIZABLE_MODEL:
        print("Error: train_on_multiple_tables requires USE_GENERALIZABLE_MODEL=True")
        return

    print("=" * 80)
    print(f"TRAINING GENERALIZABLE MODEL ON {len(dataset_indices)} TABLES")
    print("=" * 80)

    model = PermutationModel(embedding_dim=384, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ds = load_dataset("wikitablequestions", split="train")

    best_overall_reward = -float("inf")
    best_model_state = None

    for table_idx in dataset_indices:
        print(f"\n{'=' * 80}")
        print(f"Training on Table {table_idx}")
        print(f"{'=' * 80}")

        item = ds[table_idx]
        df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
        question = item["question"]
        answers = item["answers"]

        print(f"Question: {question}")
        print(f"Table shape: {df.shape}")

        # Train on this table
        reward = run_single_table_training(model, optimizer, df, question, answers, epochs=epochs_per_table, table_idx=table_idx)

        if reward > best_overall_reward:
            best_overall_reward = reward
            import copy

            best_model_state = copy.deepcopy(model.state_dict())

    # Save the final model
    if best_model_state is not None:
        model_save_path = "permutation_model_multi_table.pt"
        torch.save(best_model_state, model_save_path)
        print(f"\n{'=' * 80}")
        print(f"💾 Saved best multi-table model to {model_save_path}")
        print(f"Best overall reward: {best_overall_reward:.4f}")
        print(f"{'=' * 80}")


def run_single_table_training(model, optimizer, df, question, answers, epochs=51, table_idx=0):
    """
    Train the model on a single table for a specified number of epochs.
    Used by train_on_multiple_tables.
    """
    n = len(df)

    # Temperature schedule: start high for exploration, anneal down
    def temp_schedule(epoch):
        return max(1.0 * (0.95 ** (epoch / 10)), 0.1)

    # Entropy bonus schedule: higher early for exploration
    def entropy_weight_schedule(epoch):
        return max(0.01 * (0.99**epoch), 0.001)

    best_reward = -float("inf")
    best_model_state = None

    # Moving average baseline for stability
    baseline_ema = 0.0
    baseline_ema_alpha = 0.1

    # OPTIMIZATION: Pre-compute embeddings once (they don't change during training)
    with torch.no_grad():
        cached_embeddings = model.embed_table_and_question(df, question)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Get current temperature
        temp = temp_schedule(epoch)
        entropy_weight = entropy_weight_schedule(epoch)

        # Sample multiple permutations
        batch_P_soft = []
        batch_indices = []
        batch_permuted_dfs = []

        # OPTIMIZATION: Use cached embeddings instead of recomputing
        for _ in range(BATCH_SIZE):
            log_alpha = model(df, question, cached_embeddings=cached_embeddings)
            P_soft, indices = gumbel_sinkhorn(log_alpha, n_iters=20, temp=temp, hard=True)
            batch_P_soft.append(P_soft)
            batch_indices.append(indices)
            batch_permuted_dfs.append(df.iloc[indices.tolist()])

        # Evaluate all permutations in parallel
        rewards = asyncio.run(get_rewards_batch(batch_permuted_dfs, question, answers))

        # Update moving average baseline
        current_batch_avg = sum(rewards) / len(rewards)
        if epoch == 0:
            baseline_ema = current_batch_avg
        else:
            baseline_ema = (1 - baseline_ema_alpha) * baseline_ema + baseline_ema_alpha * current_batch_avg

        # Track best permutation
        for reward in rewards:
            if reward > best_reward:
                best_reward = reward
                import copy

                best_model_state = copy.deepcopy(model.state_dict())

        # Compute policy gradient loss
        total_policy_loss = 0.0
        total_entropy = 0.0

        for i in range(BATCH_SIZE):
            P_soft = batch_P_soft[i]
            indices = batch_indices[i]
            reward = rewards[i]

            # Compute log probabilities from the soft distribution
            log_probs = torch.log(P_soft + 1e-9)
            picked_log_probs = log_probs[torch.arange(n), indices]

            # Policy gradient loss with EMA baseline
            advantage = reward - baseline_ema
            total_policy_loss += -advantage * picked_log_probs.sum()

            # Entropy bonus
            entropy = -(P_soft * torch.log(P_soft + 1e-9)).sum()
            total_entropy += entropy

        # Average across batch
        policy_loss = total_policy_loss / BATCH_SIZE
        avg_entropy = total_entropy / BATCH_SIZE
        entropy_bonus = -entropy_weight * avg_entropy

        # Total loss
        loss = policy_loss + entropy_bonus

        # Update
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if epoch % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            print(f"Table {table_idx} | Epoch {epoch:03d} | Avg Reward: {avg_reward:+.4f} | Best: {best_reward:+.4f} | Temp: {temp:.3f}")

    return best_reward


def load_and_infer(model_path, df, question):
    """
    Load a trained PermutationModel and generate a permutation for a new table.

    Args:
        model_path: Path to saved model state dict
        df: DataFrame to permute
        question: Question to answer

    Returns:
        permuted_df, indices
    """
    model = PermutationModel(embedding_dim=384, hidden_dim=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        log_alpha = model(df, question)
        P, indices = gumbel_sinkhorn(log_alpha, n_iters=20, temp=0.01, hard=True)
        permuted_df = df.iloc[indices.tolist()]

    return permuted_df, indices.tolist()


if __name__ == "__main__":
    # Single table training example
    # run_experiment(0)

    # Multi-table training example (uncomment to use)
    train_on_multiple_tables(dataset_indices=[0, 1, 2, 3, 4], epochs_per_table=51)

    # Inference example (uncomment to use)
    ds = load_dataset("wikitablequestions", split="train")
    item = ds[15]
    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    question = item["question"]
    answers = item["answers"]

    print("\n" + "=" * 80)
    print("INFERENCE EXAMPLE - Testing Trained Model on Table 5")
    print("=" * 80)
    print(f"\nQuestion: {question}")
    print(f"Ground Truth Answers: {answers}")
    print(f"\nOriginal table shape: {df.shape}")

    # Get permutation from trained model
    permuted_df, indices = load_and_infer("permutation_model_multi_table.pt", df, question)
    print(f"\nLearned Permutation: {indices}")
    print("\nPermuted Table:")
    print(permuted_df)

    # Call LLM to get prediction and reward
    print("\n" + "-" * 80)
    print("Calling LLM with permuted table...")
    print("-" * 80)

    reward = get_llm_reward(permuted_df, question, answers)

    print(f"\nReward Score: {reward:.4f}")

    if reward >= 0.95:
        print("✓ PERFECT - Model generated correct answer with no false positives!")
    elif reward >= 0.5:
        print("✓ GOOD - Model generated partially correct answer")
    elif reward >= 0.0:
        print("⚠ PARTIAL - Model had some correct selections but also errors")
    elif reward >= -0.5:
        print("✗ POOR - Model selected wrong rows")
    else:
        print("✗ FAILED - Model did not follow format or gave empty response")

    print("\n" + "=" * 80)
