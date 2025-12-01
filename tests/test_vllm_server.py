"""
Essential tests for vLLM server components
"""
import pytest
import multiprocessing as mp
from unittest.mock import Mock, MagicMock, patch
from leap.inference.vllm_server import (
    ProcessParallelVLLM,
    VLLMWorkerProcess,
    create_generation_config,
    create_logging_config,
)


class TestGenerationConfigCreation:
    """Test configuration helper functions"""

    def test_create_generation_config_defaults(self):
        """Test creating generation config with defaults"""
        config = create_generation_config()

        assert config['use_constraints'] is True
        assert config['use_cot'] is False
        assert config['use_global_constraints'] is True
        assert 'engine_config' in config
        assert 'logging_config' in config

    def test_create_generation_config_custom(self):
        """Test creating generation config with custom values"""
        config = create_generation_config(
            use_constraints=False,
            use_cot=True,
            use_global_constraints=False,
            tensor_parallel_size=2
        )

        assert config['use_constraints'] is False
        assert config['use_cot'] is True
        assert config['use_global_constraints'] is False
        assert config['engine_config']['tensor_parallel_size'] == 2

    def test_create_logging_config_defaults(self):
        """Test creating logging config with defaults"""
        config = create_logging_config()

        assert config['enable_logging'] is True
        assert config['log_dir'] == "table_logs"
        assert config['save_readable_tables'] is True
        assert config['compress_logs'] is False
        assert config['log_format'] == "readable"

    def test_create_logging_config_custom(self):
        """Test creating logging config with custom values"""
        config = create_logging_config(
            enable_logging=False,
            log_dir="custom_logs",
            compress_logs=True
        )

        assert config['enable_logging'] is False
        assert config['log_dir'] == "custom_logs"
        assert config['compress_logs'] is True


class TestProcessParallelVLLM:
    """Test ProcessParallelVLLM initialization and basic operations"""

    def test_initialization_default_gpus(self):
        """Test initialization with default GPU allocation"""
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_generation_config()
        )

        assert server.model_id == "gpt2"
        assert server.num_workers == 2
        assert len(server.workers) == 2
        assert not server._workers_ready
        assert len(server.available_gpus) > 0

    def test_initialization_custom_gpus(self):
        """Test initialization with custom GPU allocation"""
        gpu_allocation = [0, 1, 2, 3]
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            gpu_allocation=gpu_allocation,
            generation_config=create_generation_config()
        )

        assert server.available_gpus == gpu_allocation

    def test_worker_creation(self):
        """Test that workers are created correctly"""
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=3,
            gpu_allocation=[0, 1, 2],
            generation_config=create_generation_config()
        )

        assert len(server.workers) == 3
        for i, worker in enumerate(server.workers):
            assert isinstance(worker, VLLMWorkerProcess)
            assert worker.worker_id == i
            assert worker.model_id == "gpt2"

    def test_is_ready_initially_false(self):
        """Test that server is not ready initially"""
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_generation_config()
        )

        assert not server.is_ready()

    def test_get_worker_count(self):
        """Test getting worker count"""
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=4,
            generation_config=create_generation_config()
        )

        assert server.get_worker_count() == 4

    def test_get_generation_config(self):
        """Test getting generation configuration"""
        config = create_generation_config(use_constraints=False, use_cot=True)
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            generation_config=config
        )

        retrieved_config = server.get_generation_config()
        assert retrieved_config['use_constraints'] is False
        assert retrieved_config['use_cot'] is True

    def test_logging_stats_when_disabled(self):
        """Test getting logging stats when logging is disabled"""
        config = create_generation_config()
        config['logging_config']['enable_logging'] = False

        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            generation_config=config
        )

        stats = server.get_logging_stats()
        assert stats['enabled'] is False

    def test_generate_batch_raises_when_not_ready(self):
        """Test that generate_batch raises error when workers not ready"""
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            generation_config=create_generation_config()
        )

        with pytest.raises(RuntimeError, match="Workers not ready"):
            server.generate_batch([{"question": "test"}])


class TestVLLMWorkerProcess:
    """Test VLLMWorkerProcess initialization"""

    def test_worker_initialization(self):
        """Test worker process initialization"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_generation_config(
            use_constraints=True,
            use_cot=False
        )

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config
        )

        assert worker.worker_id == 0
        assert worker.gpu_ids == [0]
        assert worker.model_id == "gpt2"
        assert worker.use_constraints is True
        assert worker.use_cot is False

    def test_worker_generation_mode_constrained(self):
        """Test generation mode string for constrained mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_generation_config(
            use_constraints=True,
            use_cot=False
        )

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config
        )

        mode = worker._get_generation_mode_string()
        assert mode == "constrained"

    def test_worker_generation_mode_cot(self):
        """Test generation mode string for chain-of-table mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_generation_config(
            use_constraints=True,
            use_cot=True
        )

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config
        )

        mode = worker._get_generation_mode_string()
        assert mode == "chain_of_table_with_constraints"

    def test_worker_generation_mode_unconstrained(self):
        """Test generation mode string for unconstrained mode"""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        generation_config = create_generation_config(
            use_constraints=False,
            use_cot=False
        )

        worker = VLLMWorkerProcess(
            worker_id=0,
            gpu_ids=[0],
            model_id="gpt2",
            input_queue=input_queue,
            output_queue=output_queue,
            generation_config=generation_config
        )

        mode = worker._get_generation_mode_string()
        assert mode == "unconstrained_with_postprocessing"


class TestTensorParallelAllocation:
    """Test GPU allocation with tensor parallelism"""

    def test_single_gpu_allocation(self):
        """Test GPU allocation without tensor parallelism"""
        config = create_generation_config(tensor_parallel_size=1)
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=3,
            gpu_allocation=[0, 1, 2, 3, 4, 5],
            generation_config=config
        )

        # Each worker should get one GPU
        assert len(server.workers) == 3

    def test_multi_gpu_tensor_parallel(self):
        """Test GPU allocation with tensor parallelism"""
        config = create_generation_config(tensor_parallel_size=2)
        server = ProcessParallelVLLM(
            model_id="gpt2",
            num_workers=2,
            gpu_allocation=[0, 1, 2, 3],
            generation_config=config
        )

        # Each worker should get 2 GPUs for tensor parallelism
        assert len(server.workers) == 2
