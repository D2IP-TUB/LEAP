from transformers import AutoTokenizer
from typing import Dict, Any


model_configs = {
    "meta-llama/Llama-2-70b-hf": {
        "log_dir": 'table_logs_llama',
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        "instruct": False

    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "log_dir": 'table_logs_llama',
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        "instruct": True

    },
    "gpt2":  {
        "log_dir": 'table_logs_gpt2',
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            # "gpu_allocation": [1, 2, 3, 4]
            "gpu_allocation": [1, 3, 5, 6]
        },
        "instruct": False
    },
    "openai/gpt-oss-120b":{
        "log_dir": 'table_logs_gpt_oss',
        "hardware_config": {
            "num_workers": 1,
            # "tensor_parallel_size": 8,
            # "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            "tensor_parallel_size": 4,
            "gpu_allocation":  [1, 2, 3, 4]
        },
        "instruct": False
    },
    "openai/gpt-oss-20b": {
        "log_dir": 'table_logs_gpt_oss_20B',
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [0, 1, 3, 5]
        },
        "instruct": False
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1":  {
        "log_dir": 'table_logs_mixtral_instruct',
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            # "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            "gpu_allocation": [0, 1, 3, 5]
            # [0, 1, 2, 3, 4, 5, 6, 7]
        },
        "instruct": True
    },
    "mistralai/Mixtral-8x7B-v0.1":  {
        "log_dir": 'table_logs_mixtral',
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            # "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            "gpu_allocation": [0, 1, 3, 5]
        },
        "instruct": False
    },
}

class ModelConfig:

    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.num_workers = model_configs[model_id]["hardware_config"]["num_workers"]
        self.tensor_parallel_size = model_configs[model_id]["hardware_config"]["tensor_parallel_size"]
        self.gpu_allocation = model_configs[model_id]["hardware_config"]["gpu_allocation"]
        self.instruct = model_configs[model_id]["instruct"]
        self.log_dir = model_configs[model_id]["log_dir"]

    def add_instruct_tokens_for_instruct_models(self, prompt, instruction_prompt):
        if self.instruct:

            message = [
                {"role": "user", "content": instruction_prompt}
            ]
            prompt += self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False).strip("<s> ")
            return prompt
        else:
            prompt += instruction_prompt
            return prompt

tokenizer_config = {
    "meta-llama/Llama-2-70b-hf": {
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
        }

    },
    "meta-llama/Llama-2-70b-chat-hf": {
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
        }

    },
    "gpt2":  {
        "closing_quote_token_ids": {
            "dot_quote_id": 526,
            "closing_bracket_quote_id": 16725,
            "colon_quote_id": 11097,
            "quote_quote_id": 15931,
            "list_close_quote_id": 30866,
            "percent_close_quote_id": 39658,
            "dash_quote_id": 21215
        }
    },
    "openai/gpt-oss-120b":{
        "closing_quote_token_ids": {
            "dot_quote_id": 3692,
            "closing_bracket_quote_id": 22445,
            "colon_quote_id": 5731,
            "quote_quote_id": 6371,
            "list_close_quote_id": 49706,
            "percent_close_quote_id": 34639,
            "dash_quote_id": 54284
        }
    },
    "openai/gpt-oss-20b": {
        "closing_quote_token_ids": {
            "dot_quote_id": 3692,
            "closing_bracket_quote_id": 22445,
            "colon_quote_id": 5731,
            "quote_quote_id": 6371,
            "list_close_quote_id": 49706,
            "percent_close_quote_id": 34639,
            "dash_quote_id": 54284
        }
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1":  {
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
        }
    },
    "mistralai/Mixtral-8x7B-v0.1":  {
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
        }
    },
}

class TokenizerConfig:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

        self.comma_id = self._get_scalar_id(",", "comma_id")
        self.list_open_id = self._get_scalar_id("[", "list_open_id")
        self.list_close_id = self._get_scalar_id("]")
        self.paren_open_id = self._get_scalar_id("(")
        self.paren_close_id = self._get_scalar_id(")")
        self.quote_id = self._get_scalar_id('"')

        closing_quote_token_ids = dict(tokenizer_config[self.tokenizer.name_or_path].get("closing_quote_token_ids", {}))

        if "closing_quote_id" not in closing_quote_token_ids:
            closing_quote_token_ids["closing_quote_id"] = self.quote_id

        closing_quotes_tokens = []
        for _, v in closing_quote_token_ids.items():
            closing_quotes_tokens.append(int(v))
        self.closing_quotes_tokens = closing_quotes_tokens
        

        self.action_tokens = {
            "select_row": self.tokenizer.encode("select_row", add_special_tokens=False),
            "select_column": self.tokenizer.encode("select_column", add_special_tokens=False),
            "end": self.tokenizer.encode("end", add_special_tokens=False),
        }

        self.token_digit_map: Dict[int, str] = {} # token_id -> "0".."9"
        self.digit_token_map: Dict[str, int] = {} # "0".."9" -> token_id
        self.digit_tokens: list[int] = [] # list[int]

        for i in range(10):
            tid = self.tokenizer.encode(str(i), add_special_tokens=False)[-1]
            tid = int(tid)
            self.digit_tokens.append(tid)
            self.token_digit_map[tid] = str(i)
            self.digit_token_map[str(i)] = tid

    def is_llama_tokenizer(self):
        return self.tokenizer.name_or_path != "gpt2"
    
    def _get_scalar_id(self, symbol: str, override_key: str | None = None) -> int:
        token_ids_overrides = tokenizer_config[self.tokenizer.name_or_path].get("token_ids", {})
        if override_key is not None and override_key in token_ids_overrides:
            val = token_ids_overrides[override_key]
            return int(val)

        tid = self.tokenizer.encode(symbol, add_special_tokens=False)[-1]
        return int(tid)