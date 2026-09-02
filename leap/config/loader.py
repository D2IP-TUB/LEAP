from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class HardwareConfig:
    num_workers: int
    tensor_parallel_size: int
    gpu_allocation: List[int]
    max_concurrent_requests: int = 16  # For continuous batching optimization
    max_model_len: int = 2048  # Maximum sequence length for the model


@dataclass(frozen=True)
class TokenizerConfig:
    is_llama_tokenizer: bool
    comma_id: int
    list_open_id: int
    list_close_id: int
    paren_open_id: int
    paren_close_id: int
    quote_id: int
    closing_quotes_tokens: List[int]
    action_tokens: Dict[str, List[int]]


@dataclass(frozen=True)
class ModelConfig:
    id: str
    instruct: bool
    log_dir: str
    hardware: HardwareConfig
    tokenizer_config: TokenizerConfig
    results_file: str


@dataclass(frozen=True)
class LoggingConfig:
    enable_logging: bool
    log_dir: str
    save_readable_tables: bool
    compress_logs: bool
    log_format: str
    max_table_chars: int


@dataclass(frozen=True)
class GenerationConfig:
    use_constraints: bool
    use_global_constraints: bool
    strategy: str = "cot"  # Strategy to use: "iterative", "cot", or "direct_query"
    constraint_backend: str = "legacy_state_machine"  # "xgrammar" or "legacy_state_machine"
    output_format: str = "function"  # "function" or "json"
    force_zero_temperature: bool = False
    sampling: Any = None  # Use Any to avoid circular import with SamplingConfig
    enabled_actions: tuple = None  # Tuple of enabled action names (immutable for frozen dataclass)


@dataclass(frozen=True)
class RunConfig:
    max_examples: Optional[int]


@dataclass(frozen=True)
class DatasetConfig:
    loader: str
    name: Optional[str] = None
    split: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    data_files: Optional[Any] = None
    path: Optional[str] = None


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    dataset: DatasetConfig
    run: RunConfig
    generation: GenerationConfig
    logging: LoggingConfig
    extractors: tuple[str, ...] = ("direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender")


def get_model_id(config_path: Path) -> str:
    """Get the model ID from config without loading everything.

    Args:
        config_path: Path to the main config file

    Returns:
        The model ID string
    """
    raw_config = _load_app_config(config_path)
    model_section = raw_config.get("model")
    if not model_section or "id" not in model_section:
        raise ValueError("Configuration must define 'model.id'")
    return model_section["id"]


def load_runtime_config(config_path: Path, tokenizer) -> AppConfig:
    """Load runtime configuration from YAML files.

    Args:
        config_path: Path to the main config file
        tokenizer: Transformers tokenizer instance for the model

    Returns:
        AppConfig with all configuration loaded and validated
    """
    raw_config = _load_app_config(config_path)

    # Configure enabled actions early, before building model config
    from leap.core.actions import REGISTRY

    generation_section = raw_config.get("generation", {})
    enabled_actions = generation_section.get("enabled_actions")
    if enabled_actions is not None:
        REGISTRY.set_enabled_actions(enabled_actions)

    model_section = raw_config.get("model")
    if not model_section or "id" not in model_section:
        raise ValueError("Configuration must define 'model.id'")

    presets_path = Path(model_section.get("presets_path", "configs/models.yaml"))
    model_presets = _load_model_presets(presets_path)
    model_config = _build_model_config(model_section, model_presets, tokenizer)

    logging_section = raw_config.get("logging", {})
    logging_config = _build_logging_config(logging_section, model_config.log_dir, model_config.id)

    dataset_section = raw_config.get("dataset")
    if not dataset_section:
        raise ValueError("Configuration must include a 'dataset' section")
    dataset_config = _build_dataset_config(dataset_section)

    run_config = RunConfig(max_examples=raw_config.get("run", {}).get("max_examples"))

    generation_section = raw_config.get("generation", {})

    # Load sampling config (import locally to avoid circular dependency)
    from leap.generation.sampling import SamplingConfig

    sampling_section = generation_section.get("sampling", {})
    sampling_config = SamplingConfig(
        enabled=sampling_section.get("enabled", False),
        n_samples=sampling_section.get("n_samples", 1),
        per_action_samples=dict(sampling_section.get("per_action_samples", {})),
        debug=sampling_section.get("debug", False),
        shuffle_invariant=sampling_section.get("shuffle_invariant", False),
    )

    generation_config = _build_generation_config(generation_section, enabled_actions, sampling_config)
    extractors = _build_extractor_config(raw_config.get("extractors", ["direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"]))

    return AppConfig(
        model=model_config,
        dataset=dataset_config,
        run=run_config,
        generation=generation_config,
        logging=logging_config,
        extractors=extractors,
    )


def load_runtime_config_tool(config_path: Path, tokenizer) -> AppConfig:
    """Load runtime configuration from YAML files.

    Args:
        config_path: Path to the main config file
        tokenizer: Transformers tokenizer instance for the model

    Returns:
        AppConfig with all configuration loaded and validated
    """
    raw_config = _load_app_config(config_path)
    # Configure enabled actions early, before building model config
    from leap.core.actions import REGISTRY

    generation_section = raw_config.get("generation", {})
    enabled_actions = generation_section.get("enabled_actions")
    if enabled_actions:
        REGISTRY.set_enabled_actions(enabled_actions)

    model_section = raw_config.get("model")
    if not model_section or "id" not in model_section:
        raise ValueError("Configuration must define 'model.id'")

    presets_path = Path(model_section.get("presets_path", "configs/models.yaml"))
    model_presets = _load_model_presets(presets_path)
    model_config = _build_model_config(model_section, model_presets, tokenizer)

    logging_section = raw_config.get("logging", {})
    logging_config = _build_logging_config(logging_section, model_config.log_dir, model_config.id)
    # dataset_section = raw_config.get("dataset")
    # if not dataset_section:
    #     raise ValueError("Configuration must include a 'dataset' section")
    dataset_config = _build_dataset_config_tool()

    run_config = RunConfig(max_examples=raw_config.get("run", {}).get("max_examples"))

    generation_section = raw_config.get("generation", {})

    # Load sampling config (import locally to avoid circular dependency)
    from leap.generation.sampling import SamplingConfig

    sampling_section = generation_section.get("sampling", {})
    sampling_config = SamplingConfig(
        enabled=sampling_section.get("enabled", False),
        n_samples=sampling_section.get("n_samples", 1),
        per_action_samples=dict(sampling_section.get("per_action_samples", {})),
        debug=sampling_section.get("debug", False),
        shuffle_invariant=sampling_section.get("shuffle_invariant", False),
    )

    generation_config = _build_generation_config(generation_section, enabled_actions, sampling_config)
    extractors = _build_extractor_config(raw_config.get("extractors", ["direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"]))

    return AppConfig(
        model=model_config,
        dataset=dataset_config,
        run=run_config,
        generation=generation_config,
        logging=logging_config,
        extractors=extractors,
    )


def update_runtime_config_dataset_tool(app_config: AppConfig, dataset_path: Path):
    dataset_config = _build_dataset_config_tool(dataset_path)
    return AppConfig(
        model=app_config.model,
        dataset=dataset_config,
        run=app_config.run,
        generation=app_config.generation,
        logging=app_config.logging,
        extractors=app_config.extractors,
    )


def _build_extractor_config(raw_extractors: Any) -> tuple[str, ...]:
    available = {"direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"}
    if not isinstance(raw_extractors, list) or not raw_extractors or not all(isinstance(name, str) for name in raw_extractors):
        raise ValueError("'extractors' must be a non-empty list of extractor names")
    duplicates = sorted({name for name in raw_extractors if raw_extractors.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate extractors are not allowed: {duplicates}")
    unknown = sorted(set(raw_extractors) - available)
    if unknown:
        raise ValueError(f"Unknown extractors {unknown}. Available extractors: {sorted(available)}")
    return tuple(raw_extractors)


def _load_app_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {config_path} must contain a mapping.")
    return data


def _build_generation_config(generation_section: Dict[str, Any], enabled_actions, sampling_config) -> GenerationConfig:
    if "batch_truncated_add_column" in generation_section:
        raise ValueError("generation.batch_truncated_add_column is no longer supported.")

    use_constraints = generation_section.get("use_constraints", False)
    constraint_backend = generation_section.get("constraint_backend", "legacy_state_machine")
    output_format = generation_section.get("output_format", "function")
    force_zero_temperature = generation_section.get("force_zero_temperature", False)

    if constraint_backend not in {"xgrammar", "legacy_state_machine"}:
        raise ValueError("generation.constraint_backend must be either 'xgrammar' or 'legacy_state_machine'.")
    if output_format not in {"function", "json"}:
        raise ValueError("generation.output_format must be 'function' or 'json'.")
    if type(force_zero_temperature) is not bool:
        raise ValueError("generation.force_zero_temperature must be a boolean.")
    if constraint_backend == "legacy_state_machine":
        output_format = "function"

    enabled_actions_tuple = tuple(enabled_actions) if enabled_actions else None

    return GenerationConfig(
        use_constraints=use_constraints,
        use_global_constraints=generation_section.get("use_global_constraints", False),
        strategy=generation_section.get("strategy", "cot"),
        constraint_backend=constraint_backend,
        output_format=output_format,
        force_zero_temperature=force_zero_temperature,
        sampling=sampling_config,
        enabled_actions=enabled_actions_tuple,
    )


def _load_model_presets(presets_path: Path) -> Dict[str, Any]:
    with open(presets_path, "r", encoding="utf-8") as presets_file:
        data = yaml.safe_load(presets_file)
    if not isinstance(data, dict):
        raise ValueError(f"Model presets file {presets_path} must define a mapping.")
    return data.get("models", data)


def _build_tokenizer_config(tokenizer_section: Dict[str, Any], tokenizer) -> TokenizerConfig:
    """Build TokenizerConfig from preset data and tokenizer instance."""
    from transformers import PreTrainedTokenizerBase

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("tokenizer must be a transformers tokenizer instance")

    is_llama_tokenizer = tokenizer_section.get("llama_tokenizer", False)

    # Helper to get token ID with optional override
    def _get_scalar_id(symbol: str, override_key: str | None = None) -> int:
        token_ids_overrides = tokenizer_section.get("token_ids", {})
        if override_key is not None and override_key in token_ids_overrides:
            return int(token_ids_overrides[override_key])
        token = tokenizer.encode(symbol, add_special_tokens=False)[-1]
        return int(token)

    comma_id = _get_scalar_id(",", "comma_id")
    list_open_id = _get_scalar_id("[", "list_open_id")
    list_close_id = _get_scalar_id("]")
    paren_open_id = _get_scalar_id("(")
    paren_close_id = _get_scalar_id(")")
    quote_id = _get_scalar_id('"')

    # Build closing quote tokens
    closing_quote_token_ids = dict(tokenizer_section.get("closing_quote_token_ids", {}))
    if "closing_quote_id" not in closing_quote_token_ids:
        closing_quote_token_ids["closing_quote_id"] = quote_id

    closing_quotes_tokens = [int(v) for v in closing_quote_token_ids.values()]

    # Build action tokens dynamically from registry
    from leap.core.actions import REGISTRY

    action_tokens = {}
    for action_name in REGISTRY.get_enabled_names():
        action_tokens[action_name] = tokenizer.encode(f"f_{action_name}", add_special_tokens=False)

    return TokenizerConfig(
        is_llama_tokenizer=is_llama_tokenizer,
        comma_id=comma_id,
        list_open_id=list_open_id,
        list_close_id=list_close_id,
        paren_open_id=paren_open_id,
        paren_close_id=paren_close_id,
        quote_id=quote_id,
        closing_quotes_tokens=closing_quotes_tokens,
        action_tokens=action_tokens,
    )


def _build_model_config(model_section: Dict[str, Any], presets: Dict[str, Any], tokenizer) -> ModelConfig:
    model_id = model_section["id"]
    preset = dict(presets.get(model_id, {}))
    if not preset:
        raise ValueError(f"No model preset found for id '{model_id}'")

    log_dir = model_section.get("log_dir", preset.get("log_dir", f"table_logs_{model_id.replace('/', '_')}"))
    instruct = model_section.get("instruct", preset.get("instruct", False))

    hardware_defaults = dict(preset.get("hardware", {}))
    hardware_overrides = model_section.get("hardware", {})
    hardware_defaults.update(hardware_overrides)

    required_hw_keys = {"num_workers", "tensor_parallel_size", "gpu_allocation"}
    missing = [key for key in required_hw_keys if key not in hardware_defaults]
    if missing:
        raise ValueError(f"Missing hardware fields {missing} for model '{model_id}'")

    hardware_config = HardwareConfig(
        num_workers=hardware_defaults["num_workers"],
        tensor_parallel_size=hardware_defaults["tensor_parallel_size"],
        gpu_allocation=list(hardware_defaults["gpu_allocation"]),
        max_concurrent_requests=hardware_defaults.get("max_concurrent_requests", 16),
        max_model_len=hardware_defaults.get("max_model_len", HardwareConfig.max_model_len),
    )

    # Build tokenizer config
    tokenizer_section = preset.get("tokenizer", {})
    if not tokenizer_section:
        raise ValueError(f"Missing tokenizer configuration for model '{model_id}'")

    tokenizer_config = _build_tokenizer_config(tokenizer_section, tokenizer)

    results_file = model_section.get("results_file", "./logs/results.jsonl")

    return ModelConfig(
        id=model_id,
        instruct=instruct,
        log_dir=log_dir,
        hardware=hardware_config,
        tokenizer_config=tokenizer_config,
        results_file=results_file,
    )


def _build_logging_config(
    raw_logging_config: Dict[str, Any],
    model_log_dir: str,
    model_id: str,
) -> LoggingConfig:
    log_dir_template = raw_logging_config.get("log_dir", "./logs/{model_log_dir}")
    log_dir = log_dir_template.format(
        model=model_id.replace("/", "_"),
        model_log_dir=model_log_dir,
    )

    return LoggingConfig(
        enable_logging=raw_logging_config.get("enable_logging", True),
        log_dir=log_dir,
        save_readable_tables=raw_logging_config.get("save_readable_tables", False),
        compress_logs=raw_logging_config.get("compress_logs", False),
        log_format=raw_logging_config.get("log_format", "readable"),
        max_table_chars=raw_logging_config.get("max_table_chars", 10000),
    )


def _build_dataset_config(raw_dataset_config: Dict[str, Any]) -> DatasetConfig:
    loader = raw_dataset_config.get("loader", "huggingface")
    return DatasetConfig(
        loader=loader,
        name=raw_dataset_config.get("name"),
        split=raw_dataset_config.get("split"),
        trust_remote_code=raw_dataset_config.get("trust_remote_code"),
        data_files=raw_dataset_config.get("data_files"),
        path=raw_dataset_config.get("path"),
    )


def _build_dataset_config_tool(path: Path = None) -> DatasetConfig:
    loader = "disk"
    return DatasetConfig(
        loader=loader,
        name="temp",
        split=None,
        trust_remote_code=None,
        data_files=None,
        path=path,
    )
