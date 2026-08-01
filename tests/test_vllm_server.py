"""
Essential tests for vLLM server components
"""

import multiprocessing as mp

import pytest

from leap.config.loader import GenerationConfig, LoggingConfig, TokenizerConfig
from leap.inference.vllm_server import ProcessParallelVLLM, VLLMWorkerProcess


# Test helper functions to create typed configs
def create_test_generation_config(
    use_constraints: bool = True,
    use_cot: bool = False,
    use_global_constraints: bool = True,
) -> GenerationConfig:
    """Helper to create GenerationConfig for tests"""
    strategy = "cot" if use_cot else "iterative"
    return GenerationConfig(
        use_constraints=use_constraints,
        strategy=strategy,
        use_global_constraints=use_global_constraints,
    )


def create_test_server(model_id: str = "gpt2", num_workers: int = 2, gpu_allocation=None, **kwargs) -> ProcessParallelVLLM:
    """Helper to create ProcessParallelVLLM with all required configs"""
    return ProcessParallelVLLM(
        model_id=model_id,
        num_workers=num_workers,
        gpu_allocation=gpu_allocation,
        generation_config=kwargs.get("generation_config", create_test_generation_config()),
        tokenizer_config=kwargs.get("tokenizer_config", create_test_tokenizer_config()),
        logging_config=kwargs.get("logging_config", create_test_logging_config()),
        generation_functions=kwargs.get("generation_functions", {}),
        tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
    )


def create_test_tokenizer_config() -> TokenizerConfig:
    """Helper to create minimal TokenizerConfig for tests"""
    return TokenizerConfig(
        is_llama_tokenizer=False,
        comma_id=11,
        list_open_id=58,
        list_close_id=60,
        paren_open_id=1306,
        paren_close_id=1572,
        quote_id=1,
        closing_quotes_tokens=[1],
        action_tokens={},
    )


def create_test_logging_config(
    enable_logging: bool = True,
    log_dir: str = "table_logs",
    save_readable_tables: bool = True,
    compress_logs: bool = False,
    log_format: str = "readable",
    max_table_chars: int = 10000,
) -> LoggingConfig:
    """Helper to create LoggingConfig for tests"""
    return LoggingConfig(
        enable_logging=enable_logging,
        log_dir=log_dir,
        save_readable_tables=save_readable_tables,
        compress_logs=compress_logs,
        log_format=log_format,
        max_table_chars=max_table_chars,
    )


class TestGenerationConfigCreation:
    """Test configuration helper functions"""

    def test_create_test_generation_config_defaults(self):
        """Test creating generation config with defaults"""
        config = create_test_generation_config()

        assert config.use_constraints is True
        assert config.strategy == "iterative"
        assert config.use_global_constraints is True

    def test_create_test_generation_config_custom(self):
        """Test creating generation config with custom values"""
        config = create_test_generation_config(use_constraints=False, use_cot=True, use_global_constraints=False)

        assert config.use_constraints is False
        assert config.strategy == "cot"
        assert config.use_global_constraints is False

    def test_create_test_logging_config_defaults(self):
        """Test creating logging config with defaults"""
        config = create_test_logging_config()

        assert config.enable_logging is True
        assert config.log_dir == "table_logs"
        assert config.save_readable_tables is True
        assert config.compress_logs is False
        assert config.log_format == "readable"

    def test_create_test_logging_config_custom(self):
        """Test creating logging config with custom values"""
        config = create_test_logging_config(enable_logging=False, log_dir="custom_logs", compress_logs=True)

        assert config.enable_logging is False
        assert config.log_dir == "custom_logs"
        assert config.compress_logs is True


class TestProcessParallelVLLM:
    """Test ProcessParallelVLLM initialization and basic operations"""

    def test_initialization_default_gpus(self):
        """Test initialization with default GPU allocation"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_test_generation_config(),
        )

        assert server.model_id == "gpt2"
        assert server.num_workers == 2
        assert len(server.workers) == 2
        assert not server._workers_ready
        assert len(server.available_gpus) > 0

    def test_initialization_custom_gpus(self):
        """Test initialization with custom GPU allocation"""
        gpu_allocation = [0, 1, 2, 3]
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            gpu_allocation=gpu_allocation,
            generation_config=create_test_generation_config(),
        )

        assert server.available_gpus == gpu_allocation

    def test_worker_creation(self):
        """Test that workers are created correctly"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=3,
            gpu_allocation=[0, 1, 2],
            generation_config=create_test_generation_config(),
        )

        assert len(server.workers) == 3
        for i, worker in enumerate(server.workers):
            assert isinstance(worker, VLLMWorkerProcess)
            assert worker.worker_id == i
            assert worker.model_id == "gpt2"

    def test_is_ready_initially_false(self):
        """Test that server is not ready initially"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_test_generation_config(),
        )

        assert not server.is_ready()

    def test_get_worker_count(self):
        """Test getting worker count"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=4,
            generation_config=create_test_generation_config(),
        )

        assert server.get_worker_count() == 4

    def test_get_generation_config(self):
        """Test getting generation configuration"""
        config = create_test_generation_config(use_constraints=False, use_cot=True)
        server = create_test_server(model_id="gpt2", num_workers=2, generation_config=config)

        retrieved_config = server.get_generation_config()
        assert retrieved_config.use_constraints is False
        assert retrieved_config.strategy == "cot"

    def test_logging_stats_when_disabled(self):
        """Test getting logging stats when logging is disabled"""
        logging_config = create_test_logging_config(enable_logging=False)

        server = create_test_server(model_id="gpt2", num_workers=2, logging_config=logging_config)

        stats = server.get_logging_stats()
        assert stats["enabled"] is False

    def test_generate_batch_raises_when_not_ready(self):
        """Test that generate_batch raises error when workers not ready"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_test_generation_config(),
        )

        with pytest.raises(RuntimeError, match="Workers not ready"):
            server.generate_batch([{"question": "test"}])

    def test_reconfigure_keeps_worker_objects_and_replaces_run_logger(self, monkeypatch, tmp_path):
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_test_generation_config(),
            logging_config=create_test_logging_config(enable_logging=False),
        )
        worker_ids = [id(worker) for worker in server.workers]
        monkeypatch.setattr(server, "workers_healthy", lambda: True)
        next_generation = GenerationConfig(
            use_constraints=False,
            use_global_constraints=False,
            strategy="cot",
            constraint_backend="xgrammar",
            output_format="json",
        )
        next_logging = create_test_logging_config(log_dir=str(tmp_path))

        server.reconfigure(next_generation, next_logging)

        assert [id(worker) for worker in server.workers] == worker_ids
        assert server.generation_config is next_generation
        assert server.logging_config is next_logging
        assert server.main_logger.log_dir == tmp_path


class TestVLLMWorkerProcess:
    """Test VLLMWorkerProcess initialization"""

    def _worker(self):
        return VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=mp.Queue(),
            output_queue=mp.Queue(),
            generation_config=create_test_generation_config(),
            tokenizer_config=create_test_tokenizer_config(),
            logging_config=create_test_logging_config(),
            generation_functions={},
            tensor_parallel_size=1,
            max_model_len=1234,
        )

    def test_apply_run_config_updates_generation_without_reloading_engine(self):
        worker = self._worker()
        engine_marker = object()
        worker.engine = engine_marker
        generation = GenerationConfig(
            use_constraints=False,
            use_global_constraints=False,
            strategy="cot",
            constraint_backend="xgrammar",
            output_format="json",
        )
        logging = create_test_logging_config(enable_logging=False)

        worker._apply_run_config(generation, logging)

        assert worker.engine is engine_marker
        assert worker.generation_config is generation
        assert worker.use_constraints is False
        assert worker.use_cot is True
        assert worker.output_format == "json"

    def test_worker_initialization(self):
        """Test worker process initialization"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_test_generation_config(use_constraints=True, use_cot=False)

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config,
            tokenizer_config=create_test_tokenizer_config(),
            logging_config=create_test_logging_config(),
            generation_functions={},
            tensor_parallel_size=1,
        )

        assert worker.worker_id == 0
        assert worker.gpu_ids == [0]
        assert worker.model_id == "gpt2"
        assert worker.use_constraints is True
        assert worker.use_cot is False

    def test_worker_resolves_vllm_012_tokenizer_and_config(self):
        """Test worker engine access for vLLM 0.12 AsyncLLM shape."""

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"

        class ModelConfig:
            max_model_len = 4096

        class VLLMConfig:
            model_config = ModelConfig()

        class Engine:
            vllm_config = VLLMConfig()

            def __init__(self):
                self.tokenizer = Tokenizer()

            def get_tokenizer(self):
                return self.tokenizer

        worker = self._worker()
        worker.engine = Engine()

        assert worker._resolve_engine_tokenizer() is worker.engine.tokenizer
        assert worker._resolve_engine_max_model_len() == 4096

    def test_worker_resolves_async_vllm_012_tokenizer(self):
        """Test worker engine access when vLLM exposes async get_tokenizer."""

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"

        class Engine:
            def __init__(self):
                self.tokenizer = Tokenizer()

            async def get_tokenizer(self):
                return self.tokenizer

        worker = self._worker()
        worker.engine = Engine()

        assert worker._resolve_engine_tokenizer() is worker.engine.tokenizer

    def test_worker_resolves_legacy_engine_tokenizer_and_config(self):
        """Test worker engine access for older AsyncLLMEngine shape."""

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"

        class TokenizerGroup:
            tokenizer = Tokenizer()

        class ModelConfig:
            max_model_len = 2048

        class InnerEngine:
            tokenizer = TokenizerGroup()
            model_config = ModelConfig()

        class Engine:
            engine = InnerEngine()

        worker = self._worker()
        worker.engine = Engine()

        assert worker._resolve_engine_tokenizer() is worker.engine.engine.tokenizer.tokenizer
        assert worker._resolve_engine_max_model_len() == 2048

    def test_worker_cleanup_shuts_down_engine_and_cuda(self, monkeypatch):
        """Test worker teardown releases vLLM and CUDA resources."""
        calls = []

        class Engine:
            def shutdown(self):
                calls.append("engine_shutdown")

        worker = self._worker()
        worker.engine = Engine()
        worker.tokenizer = object()

        monkeypatch.setattr(
            "vllm.distributed.parallel_state.cleanup_dist_env_and_memory",
            lambda: calls.append("dist_cleanup"),
        )
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.empty_cache", lambda: calls.append("empty_cache"))
        monkeypatch.setattr("torch.cuda.ipc_collect", lambda: calls.append("ipc_collect"))

        worker._cleanup_engine()

        assert worker.engine is None
        assert worker.tokenizer is None
        assert calls == ["engine_shutdown", "dist_cleanup", "empty_cache", "ipc_collect"]

    def test_worker_cleanup_supports_async_engine_shutdown(self, monkeypatch):
        """Test worker teardown handles awaitable vLLM shutdown results."""
        calls = []

        class Engine:
            async def shutdown(self):
                calls.append("engine_shutdown")

        worker = self._worker()
        worker.engine = Engine()

        monkeypatch.setattr(
            "vllm.distributed.parallel_state.cleanup_dist_env_and_memory",
            lambda: calls.append("dist_cleanup"),
        )
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        worker._cleanup_engine()

        assert calls == ["engine_shutdown", "dist_cleanup"]

    def test_worker_generation_mode_constrained(self):
        """Test generation mode string for constrained mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_test_generation_config(use_constraints=True, use_cot=False)

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config,
            tokenizer_config=create_test_tokenizer_config(),
            logging_config=create_test_logging_config(),
            generation_functions={},
            tensor_parallel_size=1,
        )

        mode = worker._get_generation_mode_string()
        assert mode == "constrained"

    def test_worker_generation_mode_cot(self):
        """Test generation mode string for chain-of-table mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_test_generation_config(use_constraints=True, use_cot=True)

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config,
            tokenizer_config=create_test_tokenizer_config(),
            logging_config=create_test_logging_config(),
            generation_functions={},
            tensor_parallel_size=1,
        )

        mode = worker._get_generation_mode_string()
        assert mode == "chain_of_table_with_constraints"

    def test_worker_generation_mode_unconstrained(self):
        """Test generation mode string for unconstrained mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_test_generation_config(use_constraints=False, use_cot=False)

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config,
            tokenizer_config=create_test_tokenizer_config(),
            logging_config=create_test_logging_config(),
            generation_functions={},
            tensor_parallel_size=1,
        )

        mode = worker._get_generation_mode_string()
        assert mode == "unconstrained_with_postprocessing"


class TestTensorParallelAllocation:
    """Test GPU allocation with tensor parallelism"""

    def test_single_gpu_allocation(self):
        """Test GPU allocation without tensor parallelism"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=3,
            gpu_allocation=[0, 1, 2, 3, 4, 5],
            tensor_parallel_size=1,
        )

        # Each worker should get one GPU
        assert len(server.workers) == 3

    def test_multi_gpu_tensor_parallel(self):
        """Test GPU allocation with tensor parallelism"""
        server = create_test_server(
            model_id="gpt2",
            num_workers=2,
            gpu_allocation=[0, 1, 2, 3],
            tensor_parallel_size=2,
        )

        # Each worker should get 2 GPUs for tensor parallelism
        assert len(server.workers) == 2
