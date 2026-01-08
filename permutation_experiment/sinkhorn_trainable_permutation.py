import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset


# --- 1. DATA MODULE ---
def load_wtq_table(index):
    dataset = load_dataset("wikitablequestions", split="train")
    item = dataset[index]
    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    return df


def get_column_data(df, col_name):
    """Converts a column to a torch tensor (numeric or alphabetical ranks)."""
    series = df[col_name]
    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.notna().all():
        return torch.tensor(numeric_series.values, dtype=torch.float32), "numerical"
    else:
        sorted_labels = sorted(series.tolist())
        ranks = torch.tensor([sorted_labels.index(line) for line in series], dtype=torch.float32)
        return ranks, "alphabetical"


# --- 2. OPERATOR MODULE ---
def stable_sinkhorn(log_alpha, n_iters=30, temp=0.1):
    P = log_alpha / temp
    for _ in range(n_iters):
        P = P - torch.logsumexp(P, dim=-1, keepdim=True)
        P = P - torch.logsumexp(P, dim=-2, keepdim=True)
    return torch.exp(P)


# --- 3. SCORING MODULE (The Blackbox) ---
class BlackboxScorer:
    @staticmethod
    def calculate_loss(permuted_values):
        """
        Example Blackbox: Maximize 'Ascendingness'.
        Instead of a target, we penalize instances where a value
        is smaller than the one before it.
        """
        # We want diffs to be positive (ascending).
        # Loss = sum of all negative 'jumps' (ReLU of negative diffs)
        diffs = permuted_values[1:] - permuted_values[:-1]
        penalty = torch.relu(-diffs).sum()
        return penalty


# --- 4. ENGINE MODULE ---
def run_optimization(df, col_name, epochs=601):
    input_data, mode = get_column_data(df, col_name)
    n = len(df)

    # Parameters
    scores = nn.Parameter(torch.randn(n, n) * 0.01)
    optimizer = optim.Adam([scores], lr=0.1)

    print(f"Optimizing '{col_name}' ({mode}) via Blackbox Score...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Generate soft permutation
        P_soft = stable_sinkhorn(scores, temp=0.1)

        # 2. Apply permutation to get the current 'guess' at the order
        permuted_values = torch.matmul(P_soft, input_data)

        # 3. Use Blackbox to find loss
        loss = BlackboxScorer.calculate_loss(permuted_values)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Blackbox Penalty: {loss.item():.4f}")

    # Final Extraction
    with torch.no_grad():
        final_P = stable_sinkhorn(scores, temp=0.01)
        indices = final_P.argmax(dim=-1).tolist()
        return df.iloc[indices]


# --- EXECUTION ---
if __name__ == "__main__":
    table_df = load_wtq_table(0)
    # Let's optimize the 'Open Cup' column using the blackbox
    result_df = run_optimization(table_df, "Playoffs")

    print("\n--- BLACKBOX OPTIMIZED TABLE ---")
    print(result_df.to_string(index=False))
