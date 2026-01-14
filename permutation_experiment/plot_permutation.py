import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

# --- CONFIG ---
ANNOTATION_DATA_PATH = "data/annotations.json"

with open(ANNOTATION_DATA_PATH) as annotated_file:
    annotations_data = json.load(annotated_file)


def infer_canonical_correct_rows(id: str):
    return annotations_data[id]


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
    scenario_counts = {"unparsable_or_empty": 0, "wrong_rows_only": 0, "correct_plus_extra_rows": 0, "exact_correct_rows": 0}

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


def correct_row_position_histogram(results_with_rewards, canonical_correct_row, top_frac=0.05):
    # sort by reward descending
    sorted_res = sorted(results_with_rewards, key=lambda r: r["reward"], reverse=True)
    k = max(1, int(len(sorted_res) * top_frac))
    top = sorted_res[:k]

    positions = []
    for r in top:
        p = r["permutation"]
        # p[new_index] = original_index
        if canonical_correct_row in p:
            pos = p.index(canonical_correct_row)  # position in permuted table
            positions.append(pos)

    print(f"\nTop-{top_frac * 100:.1f}%: {len(positions)} examples")
    for pos in range(len(p)):
        count = positions.count(pos)
        print(f"  position {pos}: {count} times")


def plot_position_mapping_heatmap(
    results_with_rewards, n, top_frac=0.1, normalize=True, subset_filter=None, save_path="pos_mapping_heatmap.png"
):
    sorted_res = sorted(results_with_rewards, key=lambda r: r["reward"], reverse=True)
    # k = max(1, int(len(sorted_res) * top_frac))
    # subset = sorted_res[:k]

    if subset_filter is not None:
        subset = [r for r in sorted_res if subset_filter(r)]
        percent = len(subset) / len(results_with_rewards) * 100
    else:
        k = max(1, int(len(sorted_res) * top_frac))
        subset = sorted_res[:k]
        percent = top_frac

    import numpy as np

    mapping = np.zeros((n, n), dtype=float)

    for r in subset:
        p = r["permutation"]
        for new_pos, orig_idx in enumerate(p):
            mapping[orig_idx, new_pos] += 1

    print(f"{save_path} data is: \n{mapping}")

    if normalize:
        col_sums = mapping.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        mapping = mapping / col_sums

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mapping, aspect="auto", origin="upper")
    label = "Probability mass" if normalize else "Count"
    plt.colorbar(im, label=label)

    plt.xlabel("Position in permuted table (after)")
    plt.ylabel("Original row index (before)")
    plt.xticks(np.arange(n), np.arange(n))
    plt.yticks(np.arange(n), np.arange(n))
    plt.title(f"Original → permuted mapping ({len(subset)} Instances - {percent:.2f}%)")
    plt.tight_layout()

    print(f"[saved] {save_path}")
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # 0. Load the WTQ instance (same one you mined on)
    ds = load_dataset("wikitablequestions", split="train")
    for i in range(0, 2):
        item = ds[i]

        df = pd.DataFrame(item["table"]["rows"], columns=item["table"]["header"])
        table_id = item["id"]

        # 1. Canonical correct rows (for the UNPERMUTED table)
        canonical_correct_rows = infer_canonical_correct_rows(table_id)
        print(f"Canonical correct rows for table with id {table_id}: {canonical_correct_rows}")

        if not canonical_correct_rows:
            raise ValueError("Could not infer canonical correct rows; please set them manually.")

        DATA_FILE = f"mined_permutations/llama3.1/table_{i}_{table_id}.json"

        # 2. Load mined permutation + generation data
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file {DATA_FILE} not found. Run the miner first.")

        with open(DATA_FILE, "r") as f:
            raw_data = json.load(f)

        print(f"Loaded {len(raw_data)} mined permutations from {DATA_FILE}")

        # 3. OFFLINE REWARD COMPUTATION
        data_with_rewards = compute_rewards(raw_data, canonical_correct_rows)

        # 🔍 Run histogram for e.g. top 5%
        correct_row_position_histogram(data_with_rewards, canonical_correct_rows[0], top_frac=0.1)

        # 🔍 Also maybe top 1%
        correct_row_position_histogram(data_with_rewards, canonical_correct_rows[0], top_frac=0.01)

        plot_position_mapping_heatmap(
            data_with_rewards,
            n=len(df),
            top_frac=1,
            normalize=False,
            subset_filter=lambda r: r["reward"] == 1,
            save_path=f"plots/{i}_succ_{table_id}.png",
        )

        plot_position_mapping_heatmap(
            data_with_rewards,
            n=len(df),
            top_frac=1,
            normalize=False,
            subset_filter=lambda r: r["reward"] < 0,
            save_path=f"plots/{i}_fail_{table_id}.png",
        )

        plot_position_mapping_heatmap(
            data_with_rewards,
            n=len(df),
            top_frac=1,
            normalize=False,
            subset_filter=lambda r: 0 < r["reward"] < 1,
            save_path=f"plots/{i}_part_{table_id}.png",
        )


if __name__ == "__main__":
    main()
