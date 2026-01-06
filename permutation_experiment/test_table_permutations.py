#!/usr/bin/env python3
"""
Standalone script to test table row selection robustness with different permutations.

This script:
1. Loads 5 instances from WikiTableQuestions dataset with ≤10 rows per table
2. Generates all permutations for each table
3. Sends async requests to vLLM with proper chat templates to select relevant rows
4. Analyzes TWO distinct metrics across permutations:
   - SEMANTIC CONSISTENCY: Does the model select the same semantic rows regardless of position?
   - POSITIONAL BIAS: Does the model prefer certain positions (e.g., always top rows)?

Usage:
1. Start vLLM server in terminal:
   vllm serve <model_name> --host 0.0.0.0 --port 8000
2. Run this script:
   python test_table_permutations.py

Testing without vLLM:
   Set MOCK_MODE = True in the configuration section to test without a server
"""

import asyncio
import itertools
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

from openai import AsyncOpenAI


# Configuration
VLLM_API_BASE = "http://localhost:8001/v1"
VLLM_MODEL_NAME = None  # Model name (None = use vLLM default, or set to specific model name)
MAX_ROWS = 10  # Maximum rows per table to keep permutations manageable
NUM_INSTANCES = 10  # Number of instances to test
MAX_CONCURRENT_REQUESTS = 32 * 4  # Limit concurrent requests
RANDOM_SEED = 42  # For reproducibility
MOCK_MODE = False  # Set to True to test without vLLM server
SHOW_SAMPLE_EVERY = 100  # Show sample prompt/response every N permutations (0 to disable)
MAX_PERMUTATIONS_PER_TABLE = None  # Maximum permutations to sample per table (None = use all permutations)

# Set random seed
random.seed(RANDOM_SEED)


def load_table_instances() -> List[Dict[str, Any]]:
    """Load WikiTableQuestions dataset and filter instances with ≤10 rows."""
    print("Loading WikiTableQuestions dataset...")

    import datasets

    try:
        dataset = datasets.load_dataset('wikitablequestions', split='train')
    except Exception as e:
        print(f"Error loading WikiTableQuestions: {e}")
        return []

    print(f"Loaded {len(dataset)} examples from WikiTableQuestions")

    filtered_instances = []
    for item in dataset:
        # WikiTableQuestions format
        table = item['table']
        header = table['header']
        rows = table['rows']
        num_rows = len(rows)

        if num_rows <= MAX_ROWS and num_rows > 0:
            question = item.get('question', 'Select relevant rows from this table.')

            # Convert to our format
            table_formatted = [header] + rows

            # No ground truth row annotations in WTQ, so we'll just track responses
            filtered_instances.append({
                'table': table_formatted,
                'question': question,
                'answers': item.get('answers', []),
                'id': item.get('id', ''),
            })

            if len(filtered_instances) >= NUM_INSTANCES:
                break

    print(f"Found {len(filtered_instances)} instances with ≤{MAX_ROWS} rows")
    return filtered_instances


def table_to_csv(header: List[str], rows: List[List[Any]]) -> str:
    """Convert table format to CSV string matching LEAP format."""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out)

    # Write header
    header_strs = [str(col) for col in header]
    writer.writerow([" "] + header_strs)

    # Write rows
    for i, row in enumerate(rows):
        row_values = [str(cell) for cell in row]
        writer.writerow([f"row {i}"] + row_values)

    return out.getvalue().strip()


def generate_permutations(num_rows: int, max_permutations: int = None) -> List[List[int]]:
    """Generate permutations of row indices, optionally sampling if total exceeds max.

    Args:
        num_rows: Number of rows to permute
        max_permutations: Maximum number of permutations to return (None = return all)

    Returns:
        List of permutations (each permutation is a list of row indices)
    """
    import math
    indices = list(range(num_rows))
    total_permutations = math.factorial(num_rows)

    # If no limit or total is within limit, generate all permutations
    if max_permutations is None or total_permutations <= max_permutations:
        perms = list(itertools.permutations(indices))
        print(f"  Generated all {len(perms)} permutations")
        return perms

    # Otherwise, randomly sample without replacement
    print(f"  Sampling {max_permutations} permutations from {total_permutations} total")

    # Use random.sample on the permutations iterator
    # For efficiency, we use reservoir sampling for large factorial values
    if total_permutations > 1000000:
        # Use reservoir sampling for very large permutation spaces
        perms = []
        for i, perm in enumerate(itertools.permutations(indices)):
            if i < max_permutations:
                perms.append(list(perm))
            else:
                # Reservoir sampling: randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < max_permutations:
                    perms[j] = list(perm)
            # Early termination after enough samples for reservoir to be unbiased
            if i >= max_permutations * 100:  # Process enough for good randomness
                break
        return perms
    else:
        # For smaller spaces, generate all and sample
        all_perms = list(itertools.permutations(indices))
        return [list(p) for p in random.sample(all_perms, max_permutations)]


def apply_permutation(rows: List[List[Any]], permutation: List[int]) -> List[List[Any]]:
    """Apply a permutation to table rows."""
    return [rows[i] for i in permutation]


def invert_permutation(permutation: List[int]) -> List[int]:
    """Compute the inverse of a permutation.

    If permutation[i] = j, then inverse[j] = i.
    This maps new positions back to original positions.

    Example: permutation = [2, 0, 1] means:
      - original row 2 is now at position 0
      - original row 0 is now at position 1
      - original row 1 is now at position 2
    Inverse = [1, 2, 0] means:
      - position 0 contains original row 2
      - position 1 contains original row 0
      - position 2 contains original row 1
    """
    inverse = [0] * len(permutation)
    for new_pos, orig_pos in enumerate(permutation):
        inverse[orig_pos] = new_pos
    return inverse


def map_to_original_indices(selected_indices: List[int], permutation: List[int]) -> List[int]:
    """Map selected row indices back to their original positions.

    Args:
        selected_indices: Row indices selected in the permuted table
        permutation: The permutation that was applied

    Returns:
        The original row indices (semantic rows)
    """
    return sorted([permutation[i] for i in selected_indices if i < len(permutation)])


def build_select_row_prompt(table_csv: str, question: str) -> List[Dict[str, str]]:
    """Build prompt for select_row action in LEAP style with proper chat template."""
    # System message with instructions
    system_message = """You are a helpful assistant that selects relevant rows from tables.
Given a table and a question, you should respond with select_row([row_indices]) containing the row numbers that are relevant to answering the question.
Only output the select_row function call, nothing else."""

    # Few-shot examples as user/assistant pairs
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

    # Current query
    current_user = f"""Table:
{table_csv}

Question: {question}"""

    # Build messages array
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example_1_user},
        {"role": "assistant", "content": example_1_assistant},
        {"role": "user", "content": example_2_user},
        {"role": "assistant", "content": example_2_assistant},
        {"role": "user", "content": current_user},
    ]

    return messages


def parse_select_row_response(response: str) -> List[int]:
    """Parse select_row response to extract row indices."""
    import re

    # Try to find select_row([...]) pattern
    pattern = r'select_row\s*\(\s*\[([^\]]*)\]\s*\)'
    match = re.search(pattern, response)

    if match:
        indices_str = match.group(1)
        try:
            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
            return indices
        except:
            pass

    # Fallback: try to extract any list of numbers
    pattern2 = r'\[([0-9,\s]+)\]'
    match2 = re.search(pattern2, response)
    if match2:
        try:
            indices = [int(x.strip()) for x in match2.group(1).split(',') if x.strip()]
            return indices
        except:
            pass

    return []


async def query_vllm(client: AsyncOpenAI, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
    """Send async request to vLLM server with chat messages.

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    if MOCK_MODE:
        # Mock response for testing without vLLM server
        await asyncio.sleep(0.01)  # Simulate network delay
        # Return a random selection for testing
        import re
        # Get the last user message to parse table
        last_message = messages[-1]['content']
        rows = re.findall(r'row \d+', last_message)
        if rows:
            max_row = len(rows) - 1
            # Randomly select 1-3 rows
            num_select = random.randint(1, min(3, max_row + 1))
            selected = random.sample(range(max_row + 1), num_select)
            return f"select_row({selected})", 0, 0
        return "select_row([0])", 0, 0

    try:
        # Get list of models to use the correct one
        if VLLM_MODEL_NAME is None:
            # Query available models and use the first one
            models = await client.models.list()
            model_name = models.data[0].id if models.data else "default"
        else:
            model_name = VLLM_MODEL_NAME

        # Use chat completions API with proper messages
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
            n=1,
        )

        # Extract token counts from usage field
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return response.choices[0].message.content.strip(), input_tokens, output_tokens
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return "", 0, 0


async def process_permutation(
    client: AsyncOpenAI,
    header: List[str],
    rows: List[List[Any]],
    permutation: List[int],
    question: str,
    semaphore: asyncio.Semaphore,
    perm_idx: int = 0,
    return_prompt: bool = False,
) -> Tuple[List[int], List[int], str, Any, List[int], int, int]:
    """Process a single permutation: apply permutation, query model, return predicted rows.

    Returns:
        Tuple of (predicted_rows_in_permuted_table, original_semantic_rows, response, messages, permutation, input_tokens, output_tokens)
    """
    async with semaphore:
        # Apply permutation to table
        permuted_rows = apply_permutation(rows, permutation)

        # Convert to CSV
        table_csv = table_to_csv(header, permuted_rows)

        # Build messages with chat template
        messages = build_select_row_prompt(table_csv, question)

        # Query model
        response, input_tokens, output_tokens = await query_vllm(client, messages)

        # Parse response (these are indices in the permuted table)
        predicted_rows_permuted = parse_select_row_response(response)

        # Map back to original indices (semantic rows)
        original_semantic_rows = map_to_original_indices(predicted_rows_permuted, permutation)

        # Return messages if requested (for samples)
        messages_to_return = messages if return_prompt else None

        return predicted_rows_permuted, original_semantic_rows, response, messages_to_return, list(permutation), input_tokens, output_tokens


async def evaluate_instance(
    client: AsyncOpenAI,
    instance: Dict[str, Any],
    instance_idx: int,
) -> Dict[str, Any]:
    """Evaluate a single instance across all permutations."""
    table = instance['table']
    header = table[0]
    rows = table[1:]
    question = instance['question']

    num_rows = len(rows)
    print(f"\nInstance {instance_idx + 1}: {num_rows} rows")
    print(f"  Question: {question[:100]}...")
    print(f"  ID: {instance.get('id', 'N/A')}")

    # Generate permutations (all or sampled based on MAX_PERMUTATIONS_PER_TABLE)
    permutations = generate_permutations(num_rows, MAX_PERMUTATIONS_PER_TABLE)

    # Process all permutations concurrently with rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
        process_permutation(
            client, header, rows, perm, question, semaphore,
            perm_idx=idx,
            return_prompt=(SHOW_SAMPLE_EVERY > 0 and idx % SHOW_SAMPLE_EVERY == 0)
        )
        for idx, perm in enumerate(permutations)
    ]

    print(f"  Processing {len(tasks)} permutations...")

    # Use tqdm progress bar if available
    if HAS_TQDM:
        results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc=f"  Instance {instance_idx + 1}"):
            results.append(await coro)
    else:
        results = await asyncio.gather(*tasks)

    # Collect and analyze results with BOTH metrics
    positional_groups = defaultdict(list)  # Group by raw position indices (positional bias)
    semantic_groups = defaultdict(list)  # Group by original semantic rows (robustness)
    samples_shown = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, (predicted_permuted, predicted_original, response, messages, permutation, input_tokens, output_tokens) in enumerate(results):
        # Aggregate token counts
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        # METRIC 1: Positional bias - group by raw selected positions
        positional_key = tuple(sorted(predicted_permuted))
        positional_groups[positional_key].append(
            {
                "permutation": permutation,
                "response": response,
                "parsed_permuted": predicted_permuted,
                "parsed_original": predicted_original,
            }
        )

        # METRIC 2: Semantic consistency - group by original semantic rows
        semantic_key = tuple(predicted_original)  # Already sorted in map_to_original_indices
        semantic_groups[semantic_key].append(
            {
                "permutation": permutation,
                "response": response,
                "parsed_permuted": predicted_permuted,
                "parsed_original": predicted_original,
            }
        )

        # Show sample prompt/response if available
        if messages and samples_shown < 3 and SHOW_SAMPLE_EVERY > 0:
            samples_shown += 1
            print(f"\n  {'='*70}")
            print(f"  SAMPLE {samples_shown} (Permutation {idx}):")
            print(f"  {'='*70}")
            print(f"  PERMUTATION: {permutation}")
            print(f"\n  MESSAGES:")
            for msg in messages:
                print(f"    [{msg['role'].upper()}]:")
                # Truncate long content
                content = msg['content']
                print(f"    {content}")
                print()
            print(f"  RESPONSE:")
            print(f"  {response}")
            print(f"\n  PARSED ROWS (in permuted table): {predicted_permuted}")
            print(f"  SEMANTIC ROWS (original indices): {predicted_original}")
            print(f"  {'=' * 70}\n")

    total = len(results)

    # METRIC 1: Positional bias analysis
    num_unique_positional = len(positional_groups)
    most_common_positional = max(positional_groups.items(), key=lambda x: len(x[1]))
    most_common_pos_rows, most_common_pos_examples = most_common_positional
    positional_consistency = len(most_common_pos_examples) / total if total > 0 else 0

    # METRIC 2: Semantic consistency analysis
    num_unique_semantic = len(semantic_groups)
    most_common_semantic = max(semantic_groups.items(), key=lambda x: len(x[1]))
    most_common_sem_rows, most_common_sem_examples = most_common_semantic
    semantic_consistency = len(most_common_sem_examples) / total if total > 0 else 0

    print(f"  Results:")
    print(f"    Total permutations: {total}")
    print(f"    Total input tokens: {total_input_tokens:,}")
    print(f"    Total output tokens: {total_output_tokens:,}")
    print(f"    Total tokens: {total_input_tokens + total_output_tokens:,}")
    print()
    print("  SEMANTIC CONSISTENCY (robustness to row order):")
    print(f"    Unique semantic responses: {num_unique_semantic}")
    print(
        f"    Most common semantic rows: {list(most_common_sem_rows)} ({len(most_common_sem_examples)}/{total} = {semantic_consistency:.2%})"
    )
    print("    Top semantic responses:")
    sorted_semantic = sorted(semantic_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (rows, examples) in enumerate(sorted_semantic[:5]):
        print(f"      {i + 1}. Original rows {list(rows)}: {len(examples)} times ({len(examples) / total:.1%})")

    print()
    print("  POSITIONAL BIAS (preference for specific positions):")
    print(f"    Unique positional responses: {num_unique_positional}")
    print(
        f"    Most common positions: {list(most_common_pos_rows)} ({len(most_common_pos_examples)}/{total} = {positional_consistency:.2%})"
    )
    print("    Top positional responses:")
    sorted_positional = sorted(positional_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (rows, examples) in enumerate(sorted_positional[:5]):
        print(f"      {i + 1}. Positions {list(rows)}: {len(examples)} times ({len(examples) / total:.1%})")

    return {
        "instance_idx": instance_idx,
        "id": instance.get("id", ""),
        "question": question,
        "num_rows": num_rows,
        "num_permutations": total,
        # Token usage metrics
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        # Semantic consistency metrics
        "semantic_consistency": {
            "num_unique_responses": num_unique_semantic,
            "most_common_response": list(most_common_sem_rows),
            "consistency_percentage": semantic_consistency,
            "response_distribution": {str(list(k)): len(v) for k, v in sorted_semantic},
            "all_responses": [
                {
                    "rows": list(k),
                    "count": len(v),
                    "percentage": len(v) / total,
                    "example_permutations": [ex["permutation"][:3] for ex in v[:3]],
                }
                for k, v in sorted_semantic
            ],
        },
        # Positional bias metrics
        "positional_bias": {
            "num_unique_responses": num_unique_positional,
            "most_common_response": list(most_common_pos_rows),
            "consistency_percentage": positional_consistency,
            "response_distribution": {str(list(k)): len(v) for k, v in sorted_positional},
            "all_responses": [
                {
                    "rows": list(k),
                    "count": len(v),
                    "percentage": len(v) / total,
                    "example_permutations": [ex["permutation"][:3] for ex in v[:3]],
                }
                for k, v in sorted_positional
            ],
        },
    }


async def main():
    """Main execution function."""
    print("=" * 80)
    print("Table Permutation Robustness Test")
    print("=" * 80)

    # Load instances
    instances = load_table_instances()

    if not instances:
        print("No suitable instances found!")
        return

    # Initialize OpenAI client for vLLM
    client = AsyncOpenAI(
        base_url=VLLM_API_BASE,
        api_key="EMPTY",  # vLLM doesn't require API key
    )

    # Evaluate each instance
    all_results = []
    for i, instance in enumerate(instances):
        result = await evaluate_instance(client, instance, i)
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_permutations = sum(r["num_permutations"] for r in all_results)
    total_input_tokens_all = sum(r["token_usage"]["total_input_tokens"] for r in all_results)
    total_output_tokens_all = sum(r["token_usage"]["total_output_tokens"] for r in all_results)
    total_tokens_all = total_input_tokens_all + total_output_tokens_all
    avg_semantic_consistency = (
        sum(r["semantic_consistency"]["consistency_percentage"] for r in all_results) / len(all_results) if all_results else 0
    )
    avg_positional_consistency = (
        sum(r["positional_bias"]["consistency_percentage"] for r in all_results) / len(all_results) if all_results else 0
    )
    total_unique_semantic = sum(r["semantic_consistency"]["num_unique_responses"] for r in all_results)
    total_unique_positional = sum(r["positional_bias"]["num_unique_responses"] for r in all_results)

    print(f"\nTotal permutations tested: {total_permutations}")
    print()
    print("TOKEN USAGE:")
    print(f"  Total input tokens: {total_input_tokens_all:,}")
    print(f"  Total output tokens: {total_output_tokens_all:,}")
    print(f"  Total tokens: {total_tokens_all:,}")
    print()
    print("SEMANTIC CONSISTENCY (robustness to row order):")
    print(f"  Average semantic consistency: {avg_semantic_consistency:.2%}")
    print(f"  Total unique semantic responses: {total_unique_semantic}")
    print()
    print("POSITIONAL BIAS (preference for specific positions):")
    print(f"  Average positional consistency: {avg_positional_consistency:.2%}")
    print(f"  Total unique positional responses: {total_unique_positional}")

    print("\nPer-instance breakdown:")
    for r in all_results:
        print(f"\n  Instance {r['instance_idx'] + 1} (ID: {r['id']}, {r['num_rows']} rows):")
        print(f"    Question: {r['question'][:80]}...")
        print(f"    Tokens: {r['token_usage']['total_input_tokens']:,} in / {r['token_usage']['total_output_tokens']:,} out / {r['token_usage']['total_tokens']:,} total")
        print(
            f"    Semantic consistency: {r['semantic_consistency']['consistency_percentage']:.2%} (most common: {r['semantic_consistency']['most_common_response']})"
        )
        print(
            f"    Positional consistency: {r['positional_bias']['consistency_percentage']:.2%} (most common: {r['positional_bias']['most_common_response']})"
        )
        print(f"    Unique semantic responses: {r['semantic_consistency']['num_unique_responses']}")
        print(f"    Unique positional responses: {r['positional_bias']['num_unique_responses']}")

    # Save results to JSON
    output_file = "permutation_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_permutations": total_permutations,
                    "token_usage": {
                        "total_input_tokens": total_input_tokens_all,
                        "total_output_tokens": total_output_tokens_all,
                        "total_tokens": total_tokens_all,
                    },
                    "semantic_consistency": {
                        "avg_consistency": avg_semantic_consistency,
                        "total_unique_responses": total_unique_semantic,
                    },
                    "positional_bias": {
                        "avg_consistency": avg_positional_consistency,
                        "total_unique_responses": total_unique_positional,
                    },
                },
                "instances": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
