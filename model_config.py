from transformers import AutoTokenizer


model_configs = {
    "meta-llama/Llama-2-70b-hf": {
        "log_dir": "table_logs_llama",
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "instruct": False,
    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "log_dir": "table_logs_llama",
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "instruct": True,
    },
    "gpt2": {
        "log_dir": "table_logs_gpt2",
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [4, 5, 6, 7],
        },
        "instruct": False,
    },
    "openai/gpt-oss-120b": {
        "log_dir": "table_logs_gpt_oss",
        "hardware_config": {
            "num_workers": 1,
            # "tensor_parallel_size": 8,
            # "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            "tensor_parallel_size": 4,
            "gpu_allocation": [1, 2, 3, 4],
        },
        "instruct": False,
    },
    "openai/gpt-oss-20b": {
        "log_dir": "table_logs_gpt_oss_20B",
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [0, 1, 3, 5],
        },
        "instruct": False,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "log_dir": "table_logs_mixtral_instruct",
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3],
        },
        "instruct": True,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "log_dir": "table_logs_mixtral",
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3],
        },
        "instruct": False,
    },
}


class ModelConfig:
    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.num_workers = model_configs[model_id]["hardware_config"]["num_workers"]
        self.tensor_parallel_size = model_configs[model_id]["hardware_config"][
            "tensor_parallel_size"
        ]
        self.gpu_allocation = model_configs[model_id]["hardware_config"][
            "gpu_allocation"
        ]
        self.instruct = model_configs[model_id]["instruct"]
        self.log_dir = model_configs[model_id]["log_dir"]

    def add_instruct_tokens_for_instruct_models(self, prompt, instruction_prompt):
        if self.instruct:
            message = [{"role": "user", "content": instruction_prompt}]
            prompt += self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            ).strip("<s> ")
            return prompt
        else:
            prompt += instruction_prompt
            return prompt
