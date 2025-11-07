from transformers import AutoTokenizer
from typing import Dict


model_configs = {
    "meta-llama/Llama-2-70b-hf": {
        "log_dir": 'table_logs_llama',
        "instruct": False

    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "log_dir": 'table_logs_llama',
        "instruct": True

    },
    "gpt2":  {
        "log_dir": 'table_logs_gpt2',
        "instruct": False
    },
    "openai/gpt-oss-120b":{
        "log_dir": 'table_logs_gpt_oss',
        "instruct": False
    },
    "openai/gpt-oss-20b": {
        "log_dir": 'table_logs_gpt_oss_20B',
        "instruct": False
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1":  {
        "log_dir": 'table_logs_mixtral_instruct',
        "instruct": True
    },
    "mistralai/Mixtral-8x7B-v0.1":  {
        "log_dir": 'table_logs_mixtral',
        "instruct": False
    },
}

class ModelConfig:

    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
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