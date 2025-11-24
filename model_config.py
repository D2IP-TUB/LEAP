
from transformers import AutoTokenizer
from typing import Dict


model_configs = {
    "meta-llama/Llama-2-70b-hf": {
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        },

    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 8,
            "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        "instruct": True

    },
    "gpt2":  {
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [1, 2, 3, 4]
        },
    },
    "openai/gpt-oss-120b":{
        "hardware_config": {
            "num_workers": 1,
            # "tensor_parallel_size": 8,
            # "gpu_allocation": [0, 1, 2, 3, 4, 5, 6, 7]
            "tensor_parallel_size": 4,
            "gpu_allocation":  [1, 2, 3, 4]
        },
    },
    "openai/gpt-oss-20b": {
        "hardware_config": {
            "num_workers": 2,
            "tensor_parallel_size": 2,
            "gpu_allocation": [0, 1, 3, 5]
        },
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1":  {
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3]
        },
    },
    "mistralai/Mixtral-8x7B-v0.1":  {
        "hardware_config": {
            "num_workers": 1,
            "tensor_parallel_size": 4,
            "gpu_allocation": [0, 1, 2, 3]
        },
    },
}

class ModelConfig:

    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.num_workers = model_configs[model_id]["hardware_config"]["num_workers"]
        self.tensor_parallel_size = model_configs[model_id]["hardware_config"]["tensor_parallel_size"]
        self.gpu_allocation = model_configs[model_id]["hardware_config"]["gpu_allocation"]