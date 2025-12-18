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
            "star_quote_id": 20605,
        },
        "token_ids": {
            "comma_id": 29892,
            "list_open_id": 29961,
        },
        "llama_tokenizer": True,
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
            "star_quote_id": 20605,
        },
        "token_ids": {
            "comma_id": 29892,
            "list_open_id": 29961,
        },
        "llama_tokenizer": True,
    },
    "gpt2": {
        "closing_quote_token_ids": {
            "dot_quote_id": 526,
            "closing_bracket_quote_id": 16725,
            "colon_quote_id": 11097,
            "quote_quote_id": 15931,
            "list_close_quote_id": 30866,
            "percent_close_quote_id": 39658,
            "dash_quote_id": 21215,
        }
    },
    "openai/gpt-oss-120b": {
        "closing_quote_token_ids": {
            "dot_quote_id": 3692,
            "closing_bracket_quote_id": 22445,
            "colon_quote_id": 5731,
            "quote_quote_id": 6371,
            "list_close_quote_id": 49706,
            "percent_close_quote_id": 34639,
            "dash_quote_id": 54284,
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
            "dash_quote_id": 54284,
        }
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "closing_quote_token_ids": {
            "closing_quote_id": 28739,
            "dot_quote_id": 611,
            "closing_bracket_quote_id": 12159,
            "colon_quote_id": 4825,
            "quote_quote_id": 2539,
            "star_quote_id": 27045,
        },
        "token_ids": {
            "comma_id": 28725,
            "list_open_id": 28792,
        },
        "llama_tokenizer": True,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "closing_quote_token_ids": {
            "closing_quote_id": 28739,
            "dot_quote_id": 611,
            "closing_bracket_quote_id": 12159,
            "colon_quote_id": 4825,
            "quote_quote_id": 2539,
            "star_quote_id": 27045,
        },
        "token_ids": {
            "comma_id": 28725,
            "list_open_id": 28792,
        },
        "llama_tokenizer": True,
    },
}


class TokenizerConfig:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        cfg = tokenizer_config[self.tokenizer.name_or_path]

        self.is_llama_tokenizer = cfg.get("llama_tokenizer", False)

        self.comma_id = self._get_scalar_id(",", "comma_id")
        self.list_open_id = self._get_scalar_id("[", "list_open_id")
        self.list_close_id = self._get_scalar_id("]")
        self.paren_open_id = self._get_scalar_id("(")
        self.paren_close_id = self._get_scalar_id(")")
        self.quote_id = self._get_scalar_id('"')

        closing_quote_token_ids = dict(cfg.get("closing_quote_token_ids", {}))

        if "closing_quote_id" not in closing_quote_token_ids:
            closing_quote_token_ids["closing_quote_id"] = self.quote_id

        closing_quotes_tokens = []
        for _, v in closing_quote_token_ids.items():
            closing_quotes_tokens.append(int(v))
        self.closing_quotes_tokens = closing_quotes_tokens

        self.action_tokens = {
            "select_row": self.tokenizer.encode("select_row", add_special_tokens=False),
            "select_column": self.tokenizer.encode(
                "select_column", add_special_tokens=False
            ),
            "end": self.tokenizer.encode("end", add_special_tokens=False),
        }

    def _get_scalar_id(self, symbol: str, override_key: str | None = None) -> int:
        token_ids_overrides = tokenizer_config[self.tokenizer.name_or_path].get(
            "token_ids", {}
        )
        if override_key is not None and override_key in token_ids_overrides:
            val = token_ids_overrides[override_key]
            return int(val)

        token = self.tokenizer.encode(symbol, add_special_tokens=False)[-1]
        return int(token)
