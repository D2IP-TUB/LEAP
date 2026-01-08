import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset


# --- 1. DATA MODULE ---
def load_wtq_item(index):
    dataset = load_dataset("wikitablequestions", split="train")
    return dataset[index]


# --- 2. LLM CLIENT MODULE ---
class LLMScorer:
    def __init__(self, server_url, target_answer):
        self.url = server_url
        self.target = target_answer

    def get_reward(self, df):
        """Calls vLLM and returns 1.0 if correct, 0.0 otherwise."""
        table_str = df.to_markdown()  # Convert table to text for LLM
        prompt = f"Based on this table:\n{table_str}\nAnswer the question concisely."

        payload = {"model": "your-model-name", "prompt": prompt, "temperature": 0}

        try:
            response = requests.post(self.url, json=payload).json()
            llm_answer = response["text"].strip()
            # Binary Reward
            return 1.0 if llm_answer.lower() == self.target.lower() else 0.0
        except Exception as e:
            print(f"Server Error: {e}")
            return 0.0


# --- 3. OPERATOR MODULE ---
def stable_sinkhorn(log_alpha, n_iters=30, temp=0.1):
    P = log_alpha / temp
    for _ in range(n_iters):
        P = P - torch.logsumexp(P, dim=-1, keepdim=True)
        P = P - torch.logsumexp(P, dim=-2, keepdim=True)
    return torch.exp(P)


# --- 4. ENGINE MODULE (REINFORCE) ---
def train_permutation(item, vllm_url, epochs=100):
    df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
    n = len(df)
    target_answer = item["answers"][0]

    scorer = LLMScorer(vllm_url, target_answer)
    scores = nn.Parameter(torch.randn(n, n) * 0.01)
    optimizer = optim.Adam([scores], lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Get probability distribution (Soft Permutation)
        P_probs = stable_sinkhorn(scores, temp=0.5)

        # 2. Sample a 'Hard' permutation based on probabilities
        # We use multinomial to sample indices for each row
        sampled_indices = torch.multinomial(P_probs, 1).flatten()

        # 3. Calculate Log Probability of this specific sample
        # (This is the 'Policy' part of Policy Gradient)
        log_prob = torch.log(P_probs[torch.arange(n), sampled_indices]).sum()

        # 4. Get Reward from LLM
        sampled_df = df.iloc[sampled_indices.tolist()]
        reward = scorer.get_reward(sampled_df)

        # 5. REINFORCE Loss: -log_prob * reward
        # If reward is 0, loss is 0. If reward is 1, we maximize log_prob
        loss = -log_prob * reward

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Reward: {reward} | Loss: {loss.item():.4f}")
            if reward > 0:
                print("Successfully found a working permutation!")

    return scores


# --- EXECUTION ---
item = load_wtq_item(0)
final_scores = train_permutation(item, "http://localhost:8000/v1/completions")
