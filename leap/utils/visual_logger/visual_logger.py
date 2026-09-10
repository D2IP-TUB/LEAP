import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import load_dataset, load_from_disk

from leap.core import Table
from leap.utils.apply_action import apply_actions

MODEL_ID = ""
LOG_DIR = ""

_project_root = Path(__file__).parent.parent.parent.parent
RESULTS_FILE = str(_project_root / "logs" / "results.jsonl")
OUTPUT_FILE = str(_project_root / "leap" / "utils" / "visual_logger" / "data.json")

config_path_env = os.environ.get("LEAP_CONFIG_PATH")
if config_path_env:
    config_path = Path(config_path_env)
else:
    config_path = _project_root / "configs" / "default.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)


def load_model_config(model_name: str = None):
    global LOG_DIR
    if not model_name:
        MODEL_ID = config_data.get("model", {}).get("id")
    else:
        MODEL_ID = model_name

    presets_path = config_data.get("model", {}).get("presets_path", "configs/models.yaml")

    if not Path(presets_path).is_absolute():
        models_path = _project_root / presets_path
    else:
        models_path = Path(presets_path)
    with open(models_path, "r", encoding="utf-8") as f:
        models_data = yaml.safe_load(f)
        models = models_data.get("models", models_data)

    model_log_dir = models[MODEL_ID].get("log_dir")

    LOG_DIR = str(_project_root / "logs" / model_log_dir)


def load_dataset_from_config(config_data: Dict[str, Any], dataset_path: Path = None):
    if not dataset_path:
        dataset_cfg = config_data.get("dataset", {})
        loader = dataset_cfg.get("loader", "huggingface").lower()
    else:
        return load_from_disk(dataset_path)
    if loader == "huggingface":
        name = dataset_cfg.get("name")
        if not name:
            raise ValueError("HuggingFace dataset loader requires 'name'")
        split = dataset_cfg.get("split")
        kwargs = {}
        if split:
            kwargs["split"] = split
        if dataset_cfg.get("trust_remote_code") is not None:
            kwargs["trust_remote_code"] = dataset_cfg["trust_remote_code"]
        return load_dataset(name, **kwargs)

    if loader == "json":
        data_files = dataset_cfg.get("data_files")
        if not data_files:
            raise ValueError("JSON dataset loader requires 'data_files'")
        split = dataset_cfg.get("split")
        return load_dataset("json", data_files=data_files, split=split)

    if loader == "disk":
        path = dataset_cfg.get("path")
        if not path:
            raise ValueError("Disk dataset loader requires 'path'")
        return load_from_disk(path)

    raise ValueError(f"Unsupported dataset loader: {loader}")


def replace_non_json_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: replace_non_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_non_json_values(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        elif math.isinf(obj):
            return "__Infinity__" if obj > 0 else "__-Infinity__"
        else:
            return obj
    else:
        return obj


def load_results_data(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    results_data = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line.strip())
                id_parts = data["id"].split("-")

                if len(id_parts) >= 2:
                    jsonl_num = int(id_parts[1])
                    log_num = jsonl_num - 1
                    request_id_base = f"req_{log_num}"
                    results_data[request_id_base] = {
                        "index": log_num,  # Store the dataset index
                        "question": data["question"],
                        "ground_truth_answers": data["ground_truth_answers"],
                        "execution_accuracy": data["execution_accuracy"],
                        "actions": data["actions"],
                        "sampling_metadata": data.get("sampling_metadata", []),
                        "metadata": data.get("metadata", {}),
                    }
        print(f"Loaded data for {len(results_data)} requests from {jsonl_path}")
    except FileNotFoundError:
        print(f"Error: {jsonl_path} not found.")
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
    return results_data


def find_log_files(log_dir: str) -> List[str]:
    try:
        files = [f for f in os.listdir(log_dir) if f.endswith("_log.json")]
        print(f"Found {len(files)} log files in {log_dir}.")
        return files
    except FileNotFoundError:
        print(f"Error: Log directory not found: {log_dir}")
        return []


def process_log_file(
    log_path: str,
    results_data: Dict[str, Dict[str, Any]],
    log_dir: str,
    dataset,
) -> List[Dict[str, Any]]:
    try:
        with open(log_path, "r", encoding="utf-8") as infile:
            content = infile.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return []

    entries = [e.strip() for e in content.split("-" * 80) if e.strip()]
    json_entries: List[Dict[str, Any]] = []

    if not entries:
        return json_entries

    first_data = json.loads(entries[0])
    full_request_id = first_data["request_id"]
    base_request_id = full_request_id.split("_")[0] + "_" + full_request_id.split("_")[1]

    replayed_tables: List[Dict[str, Any]] = []
    sampling_metadata_for_request = []
    if base_request_id in results_data:
        rd = results_data[base_request_id]
        idx = rd["index"]
        example = dataset[idx]
        initial_table = Table(
            columns=example["table"]["header"],
            rows=example["table"]["rows"],
        )

        action_dicts = rd["actions"]
        action_strings = []
        for a in action_dicts:
            name = a["action"]
            args = a.get("args", [])
            if name == "end" and not args:
                action_strings.append("end()")
            else:
                args_str = ", ".join(repr(arg) for arg in args)
                action_strings.append(f"{name}({args_str})")

        replayed = apply_actions(initial_table, action_strings)

        replayed_tables = [{"columns": t.columns, "rows": t.rows} for (_, t) in replayed]
        sampling_metadata_for_request = rd.get("sampling_metadata", [])
    for e in entries:
        try:
            entry_data = json.loads(e)
            entry_data["model_id"] = MODEL_ID
            full_request_id = entry_data["request_id"]
            step = entry_data["step"]

            base_request_id = full_request_id.split("_")[0] + "_" + full_request_id.split("_")[1]

            # Merge results_data (question, answers, execution_accuracy, etc.) into entry_data
            if base_request_id in results_data:
                rd = results_data[base_request_id]
                entry_data.update(
                    {
                        "question": rd.get("question"),
                        "ground_truth_answers": rd.get("ground_truth_answers"),
                        "execution_accuracy": rd.get("execution_accuracy"),
                        "actions": rd.get("actions"),
                    }
                )

                # Add table_data
                if step == 0:
                    # initial table from dataset
                    idx = rd["index"]
                    example = dataset[idx]
                    entry_data["table_data"] = {
                        "columns": example["table"]["header"],
                        "rows": example["table"]["rows"],
                    }
                else:
                    idx = step - 1
                    if 0 <= idx < len(replayed_tables):
                        entry_data["table_data"] = replayed_tables[idx]

                # Add sampling metadata if available
                if sampling_metadata_for_request and step > 0:
                    s_idx = step - 1
                    if 0 <= s_idx < len(sampling_metadata_for_request):
                        entry_data["sampling_for_step"] = sampling_metadata_for_request[s_idx]

            json_entries.append(entry_data)

        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON entry in {log_path}")
        except Exception as e:
            print(f"Warning: Skipping entry in {log_path} due to error: {e}")

    return json_entries


def write_output_json(data: Dict[str, Any], output_filename: str):
    try:
        data_to_dump = replace_non_json_values(data)

        print(f"Writing data to {output_filename}...")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f)

        print(f"Successfully processed data and saved to {output_filename}")
    except Exception as e:
        print(f"Error writing {output_filename}: {e}")


def main(dataset_path: Path = None, model_name: str = None):
    results_data = load_results_data(RESULTS_FILE)

    if dataset_path and model_name:
        dataset = load_dataset_from_config(config_data, dataset_path)
        load_model_config(model_name)
    else:
        dataset = load_dataset_from_config(config_data)
        load_model_config()
    log_files = find_log_files(LOG_DIR)
    if not log_files:
        print("No log files found. Exiting.")
        return

    processed_files = {}
    for f in log_files:
        print(f"--- Processing file: {f} ---")
        log_path = os.path.join(LOG_DIR, f)
        processed_files[f] = process_log_file(log_path, results_data, LOG_DIR, dataset)

    write_output_json(processed_files, OUTPUT_FILE)


if __name__ == "__main__":
    main()
