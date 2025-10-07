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
            "comma_id": 29892
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
            "comma_id": 28725
        },
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
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
            "comma_id": 28725
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

def get_logic_token_ids(tokenizer):

    logic_token_ids = {}
    closing_quote_token_ids = tokenizer_config[tokenizer.name_or_path]["closing_quote_token_ids"]
    
    comma_id = tokenizer.encode(",", add_special_tokens=False)[0]
    logic_token_ids["comma_id"] = tokenizer_config[tokenizer.name_or_path]["token_ids"]["comma_id"] if is_llama_tokenizer(tokenizer) else comma_id
    
    quote_id = tokenizer.encode('"', add_special_tokens=False)[0]
    if "closing_quote_id" not in closing_quote_token_ids:
        closing_quote_token_ids["closing_quote_id"] = quote_id

    logic_token_ids["closing_quotes_tokens"] = [val for val in closing_quote_token_ids.values()]
    
    action_tokens = {
            "select_row": tokenizer.encode("select_row", add_special_tokens=False),
            "select_column": tokenizer.encode("select_column", add_special_tokens=False),
            "end": tokenizer.encode("end", add_special_tokens=False)
    }
    logic_token_ids["action_tokens"] = action_tokens

    logic_token_ids["quote_id"] = quote_id

    logic_token_ids["paren_open_id"] = tokenizer.encode("(", add_special_tokens=False)[0]
    logic_token_ids["paren_list_open_id"] = tokenizer.encode("([", add_special_tokens=False)[0]
    logic_token_ids["list_open_id"] = tokenizer.encode("[", add_special_tokens=False)[0]

        
    
    logic_token_ids["list_close_id"] = tokenizer.encode("]", add_special_tokens=False)[0]
    logic_token_ids["paren_close_id"] = tokenizer.encode(")", add_special_tokens=False)[0]

    digit_token_map = {}
    digit_tokens = []
    for i in range(10):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        digit_tokens.append(tokens[-1])
        if tokens:
            digit_token_map[tokens[-1]] = str(i)

    logic_token_ids["digit_token_map"] = digit_token_map
    logic_token_ids["digit_tokens"] = digit_tokens

    return logic_token_ids

    

    