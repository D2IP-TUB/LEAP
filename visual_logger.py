import os
import json
import pandas as pd
import math
from typing import List, Dict, Any

LOG_DIR = "table_logs"
RESULTS_FILE = "parallel_results.jsonl"
OUTPUT_FILE = "data.json"


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
                        "question": data["question"],
                        "ground_truth_answers": data["ground_truth_answers"],
                        "execution_accuracy": data["execution_accuracy"],
                        "actions": data["actions"]
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

def find_step_csv(log_dir: str, full_request_id: str, step: int) -> str | None:
    for csv_file in os.listdir(log_dir):
        if not csv_file.startswith(full_request_id) or not csv_file.endswith('.csv'):
            continue
            
        if f"step{step:02d}" in csv_file:
            return csv_file
            
        if step < 10 and f"step{step}" in csv_file and f"step{step:02d}" not in csv_file:
            if f"step{step}_" in csv_file or f"step{step}." in csv_file or csv_file.endswith(f"step{step}.csv"):
                return csv_file
                
    return None

def read_csv_data(csv_path: str) -> Dict[str, List[Any]] | None:
    try:
        df = pd.read_csv(csv_path)
        return {
            'columns': df.columns.tolist(),
            'rows': df.values.tolist()
        }
    except Exception as e:
        print(f"Warning: Could not read CSV {csv_path}. Error: {e}")
        return None

def process_log_file(log_path: str, results_data: Dict[str, Dict[str, Any]], log_dir: str) -> List[Dict[str, Any]]:
    try:
        with open(log_path, "r", encoding="utf-8") as infile:
            content = infile.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return []

    entries = [e.strip() for e in content.split("-" * 80) if e.strip()]
    json_entries = []
    
    for e in entries:
        try:
            entry_data = json.loads(e)
            full_request_id = entry_data['request_id']
            step = entry_data['step']
            
            base_request_id = None
            base_request_id = full_request_id.split('_')[0] + "_" + full_request_id.split('_')[1] 
            
            if base_request_id and base_request_id in results_data:
                entry_data.update(results_data[base_request_id])
            
            csv_filename = find_step_csv(log_dir, full_request_id, step)
            if csv_filename:
                csv_path = os.path.join(log_dir, csv_filename)
                csv_data = read_csv_data(csv_path)
                if csv_data:
                    entry_data['csv_data'] = csv_data
            
            json_entries.append(entry_data)
            
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON entry in {log_path}")
        except Exception as e:
            print(f"Warning: Skipping entry in {log_path} due to error: {e}")
            
    return json_entries

def write_output_json(data: Dict[str, Any], output_filename: str):
    try:
        print(f"Cleaning data for JSON export...")
        data_to_dump = replace_non_json_values(data)
        
        print(f"Writing data to {output_filename}...")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f)
            
        print(f"Successfully processed data and saved to {output_filename}")
    except Exception as e:
        print(f"Error writing {output_filename}: {e}")


def main():
    results_data = load_results_data(RESULTS_FILE)
    
    log_files = find_log_files(LOG_DIR)
    if not log_files:
        print("No log files found. Exiting.")
        return

    processed_files = {}
    for f in log_files:
        print(f"--- Processing file: {f} ---")
        log_path = os.path.join(LOG_DIR, f)
        processed_files[f] = process_log_file(log_path, results_data, LOG_DIR)
    
    write_output_json(processed_files, OUTPUT_FILE)

if __name__ == "__main__":
    main()