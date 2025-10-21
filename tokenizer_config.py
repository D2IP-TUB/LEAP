from typing import List
from transformers import AutoTokenizer


tokenizer_config = {
    "meta-llama/Llama-2-70b-hf": {
        "log_dir": 'table_logs_llama',
        "closing_quote_token_ids": {
            "closing_quote_id": 29908,
            "dot_quote_id": 1213,
            "closing_bracket_quote_id": 5513,
            "colon_quote_id": 6160,
            "quote_quote_id": 5124,
            "list_close_quote_id": 18017,
            "percent_close_quote_id": 23577,
            "star_quote_id": 20605
        },
        "token_ids": {
            "comma_id": 29892,
            "list_open_id": 29961,
        },
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        }

    },
    "gpt2":  {
        "log_dir": 'table_logs_gpt2',
        "closing_quote_token_ids": {
            "dot_quote_id": 526,
            "closing_bracket_quote_id": 16725,
            "colon_quote_id": 11097,
            "quote_quote_id": 15931,
            "list_close_quote_id": 30866,
            "percent_close_quote_id": 39658,
            "dash_quote_id": 21215
        },
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [1, 2, 3, 4]
        }
    },
    "openai/gpt-oss-120b":{
        "log_dir": 'table_logs_gpt_oss',
        "closing_quote_token_ids": {
            "dot_quote_id": 3692,
            "closing_bracket_quote_id": 22445,
            "colon_quote_id": 5731,
            "quote_quote_id": 6371,
            "list_close_quote_id": 49706,
            "percent_close_quote_id": 34639,
            "dash_quote_id": 54284
        },
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        }
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1":  {
        "log_dir": 'table_logs_mixtral_instruct',
        "closing_quote_token_ids": {
            "closing_quote_id": 28739,
            "dot_quote_id": 611,
            "closing_bracket_quote_id": 12159,
            "colon_quote_id": 4825,
            "quote_quote_id": 2539,
            "star_quote_id": 27045
        },
        "token_ids": {
            "comma_id": 28725,
            "list_open_id": 28792,
        },
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            # [1, 2, 3, 4]
            # [0, 1, 2, 3, 4, 5, 6, 7]
        }
    },
    "mistralai/Mixtral-8x7B-v0.1":  {
        "log_dir": 'table_logs_mixtral',
        "closing_quote_token_ids": {
            "closing_quote_id": 28739,
            "dot_quote_id": 611,
            "closing_bracket_quote_id": 12159,
            "colon_quote_id": 4825,
            "quote_quote_id": 2539,
            "star_quote_id": 27045
        },
        "token_ids": {
            "comma_id": 28725,
            "list_open_id": 28792,
        },
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        }
    },
}

def is_llama_tokenizer(tokenizer):
     return tokenizer.name_or_path != "gpt2"

from typing import Dict, Any

def get_logic_token_ids(tokenizer) -> Dict[str, Any]:
    """
    Build a consistent, scalar-int token-id map for control symbols across tokenizers.
    This normalizes comma/list/paren/quote IDs to ints for both Llama and Mixtral.

    Assumes `tokenizer_config` (as in your snippet) is available in scope.
    """

    name = tokenizer.name_or_path
    if name not in tokenizer_config:
        raise KeyError(f"Unknown tokenizer in tokenizer_config: {name}")

    cfg = tokenizer_config[name]
    token_ids_overrides = cfg.get("token_ids", {})
    closing_quote_token_ids = dict(cfg.get("closing_quote_token_ids", {}))

    def get_scalar_id(symbol: str, override_key: str | None = None) -> int:
        # Prefer explicit override from tokenizer_config if present and int
        if override_key is not None and override_key in token_ids_overrides:
            val = token_ids_overrides[override_key]
            if not isinstance(val, int):
                raise TypeError(
                    f"Override token id for {override_key} must be int, got {type(val)}"
                )
            return int(val)

        # Fallback to tokenizer encoding (scalar int token id)
        tid = tokenizer.encode(symbol, add_special_tokens=False)[-1]
        return int(tid)

    logic_token_ids: Dict[str, Any] = {}

    # Core punctuation/symbol IDs (ALWAYS scalar ints)
    logic_token_ids["comma_id"] = get_scalar_id(",", "comma_id")
    logic_token_ids["list_open_id"] = get_scalar_id("[", "list_open_id")
    logic_token_ids["list_close_id"] = get_scalar_id("]")
    logic_token_ids["paren_open_id"] = get_scalar_id("(")
    logic_token_ids["paren_close_id"] = get_scalar_id(")")
    logic_token_ids["quote_id"] = get_scalar_id('"')

    # Ensure there is a canonical closing_quote_id in the map
    if "closing_quote_id" not in closing_quote_token_ids:
        closing_quote_token_ids["closing_quote_id"] = logic_token_ids["quote_id"]

    # Normalize all closing quote variants to ints
    closing_quotes_tokens = []
    for k, v in closing_quote_token_ids.items():
        if not isinstance(v, int):
            raise TypeError(
                f"closing_quote_token_ids[{k}] must be int, got {type(v)}"
            )
        closing_quotes_tokens.append(int(v))
    logic_token_ids["closing_quotes_tokens"] = closing_quotes_tokens

    # Action tokens (these are sequences/lists of ints; that's expected)
    action_tokens = {
        "select_row": tokenizer.encode("select_row", add_special_tokens=False),
        "select_column": tokenizer.encode("select_column", add_special_tokens=False),
        "end": tokenizer.encode("end", add_special_tokens=False),
    }
    # Sanity: ensure each action token sequence is a list of ints
    for k, seq in action_tokens.items():
        if not isinstance(seq, list) or not all(isinstance(t, int) for t in seq):
            raise TypeError(f"action_tokens[{k}] must be a list[int]")
    logic_token_ids["action_tokens"] = action_tokens

    # Digit maps: enforce single-digit mapping 0..9 to/from token ids
    token_digit_map: Dict[int, str] = {}
    digit_token_map: Dict[str, int] = {}
    digit_tokens: list[int] = []

    for i in range(10):
        tid = tokenizer.encode(str(i), add_special_tokens=False)[-1]
        tid = int(tid)
        digit_tokens.append(tid)
        token_digit_map[tid] = str(i)
        digit_token_map[str(i)] = tid

    logic_token_ids["token_digit_map"] = token_digit_map  # token_id -> "0".."9"
    logic_token_ids["digit_token_map"] = digit_token_map  # "0".."9" -> token_id
    logic_token_ids["digit_tokens"] = digit_tokens        # list[int]

    return logic_token_ids