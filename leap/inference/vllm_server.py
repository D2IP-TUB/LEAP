"""
vLLM Server Module with Integrated Logging Support

This module contains the worker process and parallel engine classes for running
vLLM inference across multiple GPUs with comprehensive table logging support.
"""

import asyncio
import gc
import inspect
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from leap.config.loader import GenerationConfig, LoggingConfig, TokenizerConfig
from leap.core import ExecutionMetrics, InferenceRequest, InferenceResult, Table

# Import the table logger
from leap.utils.table_logger import TableLogger

# Default max concurrent requests per worker for continuous batching
DEFAULT_MAX_CONCURRENT_REQUESTS = 16


@dataclass(frozen=True)
class ConfiguredInferenceRequest:
    request: InferenceRequest
    generation_config: GenerationConfig
    logging_config: LoggingConfig


def _open_progress_output():
    """Use the controlling terminal for nested suite progress while preserving stderr logs."""
    if os.environ.get("LEAP_TQDM_TO_TTY") != "1":
        return None, False
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1), True
    except OSError:
        return None, False


class VLLMWorkerProcess(mp.Process):
    """Individual worker process for vLLM inference with logging support"""

    def __init__(
        self,
        worker_id: int,
        gpu_ids: List[int],
        model_id: str,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        generation_config: GenerationConfig,
        tokenizer_config: TokenizerConfig,
        logging_config: LoggingConfig,
        generation_functions: Dict[str, Callable],
        tensor_parallel_size: int = 1,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize vLLM worker process

        Args:
            worker_id: Unique identifier for this worker
            gpu_ids: List of GPU IDs to use for this worker
            model_id: HuggingFace model identifier
            input_queue: Queue for receiving inference requests
            output_queue: Queue for sending results back
            generation_config: GenerationConfig from leap.config.loader
            tokenizer_config: TokenizerConfig from leap.config.loader
            logging_config: LoggingConfig from leap.config.loader
            generation_functions: Dict mapping strategy names to functions
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_concurrent_requests: Max concurrent requests for continuous batching
            max_model_len: Maximum sequence length for the model
        """
        super().__init__()
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids
        self.model_id = model_id
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Store typed configs directly
        self.generation_config = generation_config
        self.tokenizer_config = tokenizer_config
        self.logging_config = logging_config
        self.generation_functions = generation_functions
        self.tensor_parallel_size = tensor_parallel_size
        self.max_concurrent_requests = max_concurrent_requests
        self.configured_max_model_len = max_model_len  # Store configured value
        self.gpu_memory_utilization = gpu_memory_utilization

        # Extract commonly used fields for convenience
        self.use_constraints = generation_config.use_constraints
        self.use_cot = generation_config.strategy == "cot"
        self.use_global_constraints = generation_config.use_global_constraints
        self.constraint_backend = generation_config.constraint_backend
        self.output_format = generation_config.output_format
        self.force_zero_temperature = generation_config.force_zero_temperature

        # vLLM components (will be set after engine initialization)
        self.engine = None
        self.tokenizer = None
        self.max_model_len = None  # Will be set from engine after initialization
        self.model_loaded = mp.Event()

    def _apply_run_config(self, generation_config: GenerationConfig, logging_config: LoggingConfig) -> None:
        if generation_config != self.generation_config:
            from leap.generation.sampling import SamplingConfig, SamplingLayer
            from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer

            config = (generation_config.sampling or SamplingConfig()).for_strategy(generation_config.strategy)
            layer_type = (
                ShuffleInvariantSamplingLayer if config.shuffle_invariant and generation_config.strategy == "cot" else SamplingLayer
            )
            for generation_func in self.generation_functions.values():
                strategy = getattr(generation_func, "__self__", None)
                if strategy is not None and hasattr(strategy, "sampling_layer"):
                    strategy.sampling_layer = layer_type(config)
        self.generation_config = generation_config
        self.logging_config = logging_config
        self.use_constraints = generation_config.use_constraints
        self.use_cot = generation_config.strategy == "cot"
        self.use_global_constraints = generation_config.use_global_constraints
        self.constraint_backend = generation_config.constraint_backend
        self.output_format = generation_config.output_format
        self.force_zero_temperature = generation_config.force_zero_temperature

    def run(self):
        """Main worker process loop"""
        try:
            # Set GPU visibility for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

            # Configure enabled actions for this worker process
            # Each worker has its own REGISTRY instance that needs to be configured
            if self.generation_config.enabled_actions is not None:
                from leap.core.actions import REGISTRY

                REGISTRY.set_enabled_actions(list(self.generation_config.enabled_actions))

            generation_mode = self._get_generation_mode_string()
            print(f"Worker {self.worker_id} starting with GPUs: {self.gpu_ids}, mode: {generation_mode}")

            # Initialize logger for this worker - DON'T create separate logger
            # Workers will send logging data back to main process
            # self._init_logger()  # REMOVED

            # Initialize engine
            self._init_engine()

            # Signal model loaded
            self.model_loaded.set()
            print(f"Worker {self.worker_id}: Model loaded successfully")

            # Send ready signal
            self.output_queue.put(("ready", self.worker_id, None))

            # Process requests
            asyncio.run(self._process_requests())

        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
            self.output_queue.put(("error", self.worker_id, str(e)))
        finally:
            self._cleanup_engine()

    def _get_generation_mode_string(self) -> str:
        """Get descriptive string for generation mode"""
        format_prefix = f"{self.output_format}_" if self.output_format != "function" else ""
        if self.use_cot:
            constraint_desc = "with_constraints" if self.use_constraints else "without_constraints"
            return f"{format_prefix}chain_of_table_{constraint_desc}"
        elif self.use_constraints:
            return f"{format_prefix}constrained"
        else:
            return f"{format_prefix}unconstrained_with_postprocessing"

    def effective_temperature(self, temperature: float) -> float:
        return 0.0 if self.force_zero_temperature else temperature

    def _init_engine(self):
        """Initialize the vLLM engine"""
        print(f"Worker {self.worker_id}: Starting model loading...")
        start_time = time.time()

        # Use typed config instead of dict
        # Ensure max_num_batched_tokens is at least as large as max_model_len
        max_num_batched_tokens = max(8192, self.configured_max_model_len)

        engine_args = AsyncEngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=self.configured_max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=32,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = self._resolve_engine_tokenizer()
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = self._resolve_engine_max_model_len()

        load_time = time.time() - start_time
        print(f"Worker {self.worker_id}: Model loaded in {load_time:.2f} seconds, max_model_len={self.max_model_len}")

    def _resolve_engine_tokenizer(self):
        """Return the tokenizer across vLLM engine API versions."""
        if hasattr(self.engine, "get_tokenizer"):
            tokenizer = self.engine.get_tokenizer()
            if inspect.isawaitable(tokenizer):
                tokenizer = asyncio.run(tokenizer)
            return tokenizer

        tokenizer_owner = getattr(self.engine, "engine", None)
        if tokenizer_owner is not None and hasattr(tokenizer_owner, "tokenizer"):
            tokenizer_group = tokenizer_owner.tokenizer
            return getattr(tokenizer_group, "tokenizer", tokenizer_group)

        tokenizer = getattr(self.engine, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        raise AttributeError("Unable to resolve tokenizer from vLLM engine.")

    def _resolve_engine_max_model_len(self) -> int:
        """Return max model length across vLLM engine API versions."""
        engine_core = getattr(self.engine, "engine", None)
        model_config = getattr(engine_core, "model_config", None)
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_model_len is not None:
            return max_model_len

        vllm_config = getattr(self.engine, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_model_len is not None:
            return max_model_len

        return self.configured_max_model_len

    def _cleanup_engine(self):
        """Release vLLM and CUDA resources before the worker process exits."""
        engine = self.engine
        self.engine = None
        self.tokenizer = None

        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                try:
                    result = shutdown()
                    if inspect.isawaitable(result):
                        asyncio.run(result)
                except Exception as e:
                    print(f"Worker {self.worker_id}: vLLM engine shutdown failed: {e}")

        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        except Exception as e:
            print(f"Worker {self.worker_id}: vLLM distributed cleanup failed: {e}")

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Worker {self.worker_id}: CUDA cache cleanup failed: {e}")

        gc.collect()

    async def _process_requests(self):
        """Process requests until the worker receives a shutdown signal."""
        await self._process_request_loop()

    async def _process_request_loop(self):
        """Process incoming requests with concurrent batching support."""
        state_machines = {}
        active_tasks = {}  # request_id -> (task, request_id)
        max_concurrent = self.max_concurrent_requests
        shutdown_requested = False

        print(f"Worker {self.worker_id}: Processing with max_concurrent={max_concurrent} for continuous batching")

        while True:
            try:
                # Try to fill up to max_concurrent tasks
                # Use a more aggressive refill strategy to prevent starvation
                slots_available = max_concurrent - len(active_tasks)
                if slots_available > 0 and not shutdown_requested:
                    # Try to fill multiple slots at once to reduce queue contention
                    filled = 0
                    for _ in range(slots_available):
                        try:
                            queued_request = self.input_queue.get_nowait()

                            if queued_request is None:  # Shutdown signal
                                shutdown_requested = True
                                break

                            if isinstance(queued_request, ConfiguredInferenceRequest):
                                self._apply_run_config(queued_request.generation_config, queued_request.logging_config)
                                request = queued_request.request
                            else:
                                request = queued_request

                            # Create async task for this request (non-blocking)
                            task = asyncio.create_task(self._process_single_request(request, state_machines))
                            active_tasks[request.request_id] = (task, request.request_id)
                            filled += 1

                        except queue.Empty:
                            break

                    # Only log when we actually filled slots (less verbose)
                    # Comment this out for even less logging noise
                    # if filled > 0:
                    #     print(f"Worker {self.worker_id}: Filled {filled} slots, now {len(active_tasks)}/{max_concurrent} active")

                # If no active tasks and shutdown requested, exit
                if not active_tasks and shutdown_requested:
                    break

                # If we have active tasks, wait for at least one to complete
                if active_tasks:
                    # Wait for first completion (or timeout to check for new work)
                    done, pending = await asyncio.wait(
                        [task for task, _ in active_tasks.values()],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.05,  # Reduced timeout for faster queue checking
                    )

                    # Process completed tasks
                    for completed_task in done:
                        # Find which request_id this task belongs to
                        completed_request_id = None
                        for req_id, (task, _) in active_tasks.items():
                            if task == completed_task:
                                completed_request_id = req_id
                                break

                        if completed_request_id:
                            try:
                                result = completed_task.result()
                                # Send result back
                                self.output_queue.put(("result", completed_request_id, result))
                            except Exception as e:
                                print(f"Worker {self.worker_id} task error for {completed_request_id}: {e}")
                                # Send error result
                                error_result = InferenceResult(
                                    action_history=[],
                                    final_table=Table(columns=[], rows=[]),
                                    execution_metrics=ExecutionMetrics(
                                        execution_accuracy=0.0,
                                        answer_found_in_final=False,
                                        answer_found_in_original=False,
                                        terminated_properly=False,
                                        matched_answers_final=[],
                                        matched_answers_original=[],
                                        num_actions=0,
                                        execution_error=str(e),
                                    ),
                                    request_id=completed_request_id,
                                    question="",
                                    ground_truth_answers=[],
                                )
                                self.output_queue.put(("result", completed_request_id, error_result))

                            # Clean up state machines for this request
                            to_delete = [key for key in state_machines if key.startswith(completed_request_id)]
                            for key in to_delete:
                                del state_machines[key]

                            # Remove from active tasks
                            del active_tasks[completed_request_id]
                else:
                    # No active tasks, try to get some work
                    await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Worker {self.worker_id} error in main loop: {e}")

    async def _process_single_request(self, request: InferenceRequest, state_machines: Dict) -> InferenceResult:
        """
        Process a single request using the configured generation function with logging

        Args:
            request: InferenceRequest object from queue
            state_machines: Shared state machines for constraint processing

        Returns:
            InferenceResult object
        """
        try:
            # Get the appropriate generation function based on strategy config
            strategy = self.generation_config.strategy
            function_name = f"{self.output_format}_{strategy}_generation"
            generation_func = self.generation_functions.get(function_name) or self.generation_functions.get(strategy + "_generation")

            if not generation_func:
                raise ValueError(f"No generation function configured for strategy: {strategy}")

            # Call the generation function with typed request
            result: InferenceResult = await generation_func(
                request=request,
                worker=self,
                state_machines=state_machines,
                logging_callback=self._send_log_entry,  # Send logs back to main process
            )

            # Return the typed result object directly
            return result

        except asyncio.CancelledError:
            # Task was cancelled - this should not crash the worker
            # Return error result instead of propagating
            print(f"Worker {self.worker_id} request {request.request_id} was cancelled")
            return InferenceResult(
                action_history=[],
                final_table=Table(columns=[], rows=[]),
                execution_metrics=ExecutionMetrics(
                    execution_accuracy=0.0,
                    answer_found_in_final=False,
                    answer_found_in_original=False,
                    terminated_properly=False,
                    matched_answers_final=[],
                    matched_answers_original=[],
                    num_actions=0,
                    execution_error="Request cancelled",
                ),
                request_id=request.request_id,
                question=request.question,
                ground_truth_answers=request.ground_truth_answers,
            )
        except Exception as e:
            # If an error occurs during generation, create a failed result
            print(f"Worker {self.worker_id} error processing request {request.request_id}: {e}")

            # Return an error result object
            return InferenceResult(
                action_history=[],
                final_table=Table(columns=[], rows=[]),
                execution_metrics=ExecutionMetrics(
                    execution_accuracy=0.0,
                    answer_found_in_final=False,
                    answer_found_in_original=False,
                    terminated_properly=False,
                    matched_answers_final=[],
                    matched_answers_original=[],
                    num_actions=0,
                    execution_error=str(e),
                ),
                request_id=request.request_id,
                question=request.question,
                ground_truth_answers=request.ground_truth_answers,
            )

    def _send_log_entry(
        self,
        request_id: str,
        step: int,
        action: str,
        table: Dict[str, Any],
        success: bool = True,
        failure_type: str = None,
        generation_mode: str = None,
    ):
        """Send logging data back to main process via output queue"""
        if self.logging_config.enable_logging:
            log_data = {
                "request_id": request_id,
                "step": step,
                "action": action,
                "table": table,
                "success": success,
                "failure_type": failure_type,
                "generation_mode": generation_mode,
                "worker_id": self.worker_id,
            }
            # Send log entry back to main process
            self.output_queue.put(("log_entry", request_id, log_data))

    async def generate_text(self, prompt: str, request_id: str, sampling_params: SamplingParams) -> str:
        """
        Generate text using the vLLM engine

        Args:
            prompt: Input prompt
            request_id: Unique request identifier
            sampling_params: Sampling parameters for generation

        Returns:
            Generated text

        Raises:
            Exception: Re-raises exceptions from vLLM to allow proper error handling upstream
        """
        try:
            result_generator = self.engine.generate(prompt, sampling_params, request_id)

            final_result = None
            async for result in result_generator:
                final_result = result

            if final_result and final_result.outputs:
                return final_result.outputs[0].text.strip()
            else:
                return ""
        except asyncio.CancelledError:
            # CancelledError should propagate for proper asyncio cancellation
            # Abort the request but don't suppress the cancellation
            try:
                await self.engine.abort(request_id)
            except Exception:
                pass
            raise
        except Exception:
            # Abort the request to clean up vLLM engine state
            # This prevents the engine from getting stuck on failed requests
            try:
                await self.engine.abort(request_id)
            except Exception:
                pass  # Ignore abort errors
            # Re-raise so _process_single_request can handle it gracefully
            raise


class ProcessParallelVLLM:
    """Process-based parallel vLLM engine with integrated logging"""

    def __init__(
        self,
        model_id: str,
        num_workers: int,
        gpu_allocation: Optional[List[int]] = None,
        generation_config: GenerationConfig = None,
        tokenizer_config: TokenizerConfig = None,
        logging_config: LoggingConfig = None,
        generation_functions: Dict[str, Callable] = None,
        tensor_parallel_size: int = 1,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize parallel vLLM engine

        Args:
            model_id: HuggingFace model identifier
            num_workers: Number of worker processes
            gpu_allocation: Optional list of GPU IDs to use (auto-detected if None)
            generation_config: GenerationConfig from leap.config.loader
            tokenizer_config: TokenizerConfig from leap.config.loader
            logging_config: LoggingConfig from leap.config.loader
            generation_functions: Dict mapping strategy names to functions
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_concurrent_requests: Max concurrent requests per worker for continuous batching
            max_model_len: Maximum sequence length for the model
        """
        self.model_id = model_id
        self.num_workers = num_workers
        self.max_concurrent_requests = max_concurrent_requests
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        # Auto-detect GPUs if not specified
        if gpu_allocation is None:
            import subprocess

            try:
                result = subprocess.run(
                    ["nvidia-smi", "--list-gpus"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                num_gpus = len(result.stdout.strip().split("\n"))
                self.available_gpus = list(range(num_gpus))
            except Exception:
                # Fallback if nvidia-smi not available
                self.available_gpus = list(range(8))  # Assume 8 GPUs max
        else:
            self.available_gpus = gpu_allocation

        # Store typed configs
        self.generation_config = generation_config
        self.tokenizer_config = tokenizer_config
        self.logging_config = logging_config
        self.generation_functions = generation_functions or {}
        self.tensor_parallel_size = tensor_parallel_size

        # Process management
        self.workers = []
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self._workers_ready = False

        # Initialize main logger for coordination
        self.main_logger = None
        self._init_main_logger()

        # Create worker processes
        self._create_workers()

    def _init_main_logger(self):
        """Initialize main logger for coordination and summary"""
        if self.logging_config.enable_logging:
            self.main_logger = TableLogger(
                log_dir=self.logging_config.log_dir,
                enable_logging=True,
                save_readable_tables=self.logging_config.save_readable_tables,
                compress_logs=self.logging_config.compress_logs,
                log_format=self.logging_config.log_format,
                max_table_chars=self.logging_config.max_table_chars,
            )
            print(f"Main logger initialized: {self.logging_config.log_dir}")

    def _create_workers(self):
        """Create worker processes with GPU allocation"""

        if self.tensor_parallel_size > 1:
            print(f"Spawning workers with {self.tensor_parallel_size} GPUs allocated.")

        for worker_id in range(self.num_workers):
            if self.tensor_parallel_size > 1:
                first_gpu = worker_id * self.tensor_parallel_size
                final_gpu = first_gpu + self.tensor_parallel_size
                gpu_ids = self.available_gpus[first_gpu:final_gpu]
                print(f"Assigning GPUs {gpu_ids} to worker {worker_id}")
            else:
                gpu_ids = [self.available_gpus[worker_id % len(self.available_gpus)]]
                print(f"Assigning GPU {gpu_ids} to worker {worker_id}")

            worker = VLLMWorkerProcess(
                worker_id=worker_id,
                gpu_ids=gpu_ids,
                model_id=self.model_id,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                generation_config=self.generation_config,
                tokenizer_config=self.tokenizer_config,
                logging_config=self.logging_config,
                generation_functions=self.generation_functions,
                tensor_parallel_size=self.tensor_parallel_size,
                max_concurrent_requests=self.max_concurrent_requests,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            self.workers.append(worker)

    def start_workers(self, timeout: int = 300) -> bool:
        """
        Start all worker processes and wait for models to load

        Args:
            timeout: Timeout in seconds for model loading

        Returns:
            True if all workers started successfully, False otherwise
        """
        print(f"Starting {self.num_workers} worker processes...")

        # Determine generation mode for logging
        generation_mode = self._get_generation_mode_description()
        print(f"Generation mode: {generation_mode}")

        for worker in self.workers:
            worker.start()

        # Wait for all workers to load models and be ready
        ready_count = 0
        start_time = time.time()

        while ready_count < self.num_workers:
            try:
                msg_type, worker_id, data = self.output_queue.get(timeout=timeout)
                if msg_type == "ready":
                    print(f"Worker {worker_id} is ready")
                    ready_count += 1
                elif msg_type == "error":
                    print(f"Worker {worker_id} failed: {data}")
            except queue.Empty:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    print(f"Timeout waiting for workers to load models ({timeout}s)")
                    break

        self._workers_ready = ready_count == self.num_workers

        if self._workers_ready:
            print(f"All {self.num_workers} workers are ready")
        else:
            print(f"Only {ready_count}/{self.num_workers} workers ready")

        return self._workers_ready

    def _get_generation_mode_description(self) -> str:
        """Get description of current generation mode"""
        use_cot = self.generation_config.strategy == "cot"
        use_constraints = self.generation_config.use_constraints

        if use_cot:
            return "Chain-of-Table (dynamic_plan + generate_args)"
        elif use_constraints:
            return "Constrained generation"
        else:
            return "Unconstrained generation with post-processing"

    def reconfigure(self, generation_config: GenerationConfig, logging_config: LoggingConfig) -> None:
        """Select settings and a fresh logger for the next batch without recreating workers."""
        if not self.workers_healthy():
            raise RuntimeError("Cannot reconfigure an unhealthy vLLM worker pool.")
        self.generation_config = generation_config
        self.logging_config = logging_config
        self.main_logger = None
        self._init_main_logger()

    def finish_run(self) -> None:
        """Finalize the active run's logs without shutting down the model workers."""
        if self.main_logger:
            self.write_summary_report(self._get_generation_mode_description())

    def workers_healthy(self) -> bool:
        started_workers = [worker for worker in self.workers if worker.pid is not None]
        return self._workers_ready and len(started_workers) == self.num_workers and all(worker.is_alive() for worker in started_workers)

    def generate_batch(self, requests: List[InferenceRequest], timeout_per_request: int = 180) -> List[InferenceResult]:
        """
        Generate responses for batch of requests

        Args:
            requests: List of InferenceRequest objects
            timeout_per_request: Timeout per request in seconds

        Returns:
            List of InferenceResult objects in same order as requests
        """
        if not self._workers_ready:
            raise RuntimeError("Workers not ready. Call start_workers() first.")

        # Send all requests to queue using their existing stable IDs
        request_ids = []
        for request in requests:
            request_id = request.request_id
            request_ids.append(request_id)

            # Put the typed object directly in the queue
            self.input_queue.put(
                ConfiguredInferenceRequest(
                    request=request,
                    generation_config=self.generation_config,
                    logging_config=self.logging_config,
                )
            )

        # Collect results
        results = {}
        completed = 0
        total_requests = len(request_ids)
        progress_file, close_progress_file = _open_progress_output()
        try:
            with tqdm(
                total=total_requests,
                desc="Questions",
                unit="question",
                dynamic_ncols=True,
                position=int(os.environ.get("LEAP_TQDM_POSITION", "0")),
                leave=os.environ.get("LEAP_TQDM_LEAVE", "1") == "1",
                file=progress_file,
            ) as progress:
                while completed < total_requests:
                    try:
                        msg_type, req_id, data = self.output_queue.get(timeout=timeout_per_request)
                        if msg_type == "result":
                            # data is already an InferenceResult object
                            results[req_id] = data
                            completed += 1
                            progress.update(1)
                        elif msg_type == "log_entry":
                            # Handle log entries from workers
                            if self.main_logger:
                                log_data = data
                                self.main_logger.log_table_state(
                                    request_id=log_data["request_id"],
                                    step=log_data["step"],
                                    action=log_data["action"],
                                    table=log_data["table"],
                                    success=log_data["success"],
                                    failure_type=log_data["failure_type"],
                                    generation_mode=log_data["generation_mode"],
                                )
                        elif msg_type == "error":
                            tqdm.write(f"Error for request {req_id}: {data}", file=progress_file)
                            # Create error result object
                            results[req_id] = InferenceResult(
                                action_history=[],
                                final_table=Table(columns=[], rows=[]),
                                execution_metrics=ExecutionMetrics(
                                    execution_accuracy=0.0,
                                    answer_found_in_final=False,
                                    answer_found_in_original=False,
                                    terminated_properly=False,
                                    matched_answers_final=[],
                                    matched_answers_original=[],
                                    num_actions=0,
                                    execution_error=str(data),
                                ),
                                request_id=req_id,
                                question="",
                                ground_truth_answers=[],
                            )
                            completed += 1
                            progress.update(1)
                    except queue.Empty:
                        tqdm.write(f"Timeout waiting for results (completed {completed}/{total_requests})", file=progress_file)
                        break
        finally:
            if close_progress_file:
                progress_file.close()

        # Return results in original order, with per-request error results for missing ones.
        ordered_results = []
        for req_id in request_ids:
            result = results.get(req_id)
            if result is None:
                result = InferenceResult(
                    action_history=[],
                    final_table=Table(columns=[], rows=[]),
                    execution_metrics=ExecutionMetrics(
                        execution_accuracy=0.0,
                        answer_found_in_final=False,
                        answer_found_in_original=False,
                        terminated_properly=False,
                        matched_answers_final=[],
                        matched_answers_original=[],
                        num_actions=0,
                        execution_error="Request not completed",
                    ),
                    request_id=req_id,
                    question="",
                    ground_truth_answers=[],
                )
            if self.main_logger:
                self.main_logger.record_inference_result(result)
            ordered_results.append(result)

        return ordered_results

    def create_summary_report(self, generation_mode: str = None) -> Dict[str, Any]:
        """Create summary report using main logger"""
        if self.main_logger:
            return self.main_logger.create_summary_report(generation_mode)
        else:
            return {"error": "Logging not enabled"}

    def write_summary_report(self, generation_mode: str = None) -> None:
        """Write comprehensive summary report"""
        if self.main_logger:
            self.main_logger.write_summary_report(generation_mode)
        else:
            print("Warning: Logging not enabled, cannot write summary report")

    def analyze_request_logs(self, request_id: str) -> None:
        """Analyze logs for a specific request"""
        if self.main_logger:
            self.main_logger.analyze_table_logs(request_id)
        else:
            print("Warning: Logging not enabled, cannot analyze logs")

    def shutdown(self, *, write_summary: bool = True):
        """Shutdown all worker processes"""
        print("Shutting down workers...")

        # Send shutdown signals
        started_workers = [worker for worker in self.workers if worker.pid is not None]
        for _ in started_workers:
            self.input_queue.put(None)

        # Wait for workers to finish
        for worker in started_workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Force terminating worker {worker.worker_id}")
                worker.terminate()
                worker.join()

        self._workers_ready = False

        # Final summary report if logging enabled
        if write_summary and self.main_logger:
            print("\nGenerating final summary report...")
            generation_mode = self._get_generation_mode_description()
            self.write_summary_report(generation_mode)

        self._close_queues()

    def _close_queues(self):
        """Close multiprocessing queues after workers have stopped."""
        for process_queue in (self.input_queue, self.output_queue):
            try:
                process_queue.close()
                process_queue.join_thread()
            except Exception:
                pass

    def is_ready(self) -> bool:
        """Check if all workers are ready"""
        return self._workers_ready

    def get_worker_count(self) -> int:
        """Get number of workers"""
        return self.num_workers

    def get_generation_config(self) -> GenerationConfig:
        """Get current generation configuration"""
        return self.generation_config

    def get_logging_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        if self.main_logger:
            return self.main_logger.get_logging_stats()
        else:
            return {"enabled": False}


# Example usage and configuration helpers
def setup_standard_vllm_server(
    model_id: str,
    num_workers: int = 4,
    gpu_allocation: List[int] = None,
    use_constraints: bool = True,
    use_cot: bool = False,
    enable_logging: bool = True,
    log_dir: str = "table_logs",
    generation_functions: Dict[str, Callable] = None,
) -> ProcessParallelVLLM:
    """
    Setup a standard vLLM server with common configuration and logging

    Args:
        model_id: HuggingFace model identifier
        num_workers: Number of worker processes
        gpu_allocation: List of GPU IDs to use
        use_constraints: Whether to use constrained generation
        use_cot: Whether to use Chain-of-Table approach
        enable_logging: Whether to enable comprehensive logging
        log_dir: Directory for log files
        generation_functions: Dictionary mapping strategy names to generation functions

    Returns:
        Configured ProcessParallelVLLM instance
    """
    from dataclasses import replace
    from pathlib import Path

    from transformers import AutoTokenizer

    from leap.config.loader import get_model_id, load_runtime_config

    # Load typed configs from the standard config system
    config_path = Path("configs/default.yaml")
    model_id_from_config = get_model_id(config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id_from_config)
    app_config = load_runtime_config(config_path, tokenizer)

    # Override specific settings using dataclass replace (since they're frozen)
    # Convert use_cot to strategy
    strategy = "cot" if use_cot else app_config.generation.strategy
    generation_config = replace(
        app_config.generation,
        use_constraints=use_constraints,
        strategy=strategy,
    )

    logging_config = replace(app_config.logging, enable_logging=enable_logging, log_dir=log_dir)

    return ProcessParallelVLLM(
        model_id=model_id,
        num_workers=num_workers,
        gpu_allocation=gpu_allocation,
        generation_config=generation_config,
        tokenizer_config=app_config.model.tokenizer_config,
        logging_config=logging_config,
        generation_functions=generation_functions or {},
        tensor_parallel_size=app_config.model.hardware.tensor_parallel_size,
        max_model_len=app_config.model.hardware.max_model_len,
        gpu_memory_utilization=app_config.model.hardware.gpu_memory_utilization,
    )


if __name__ == "__main__":
    # Example usage with logging
    print("vLLM Server Module with Logging - Example Usage")

    # Create a server with logging enabled
    server = setup_standard_vllm_server(
        model_id="gpt2",
        num_workers=2,
        use_constraints=False,
        use_cot=False,
        enable_logging=True,
        log_dir="example_logs",
    )

    try:
        # Start workers
        if server.start_workers():
            print("Server started successfully!")
            print(f"Logging stats: {server.get_logging_stats()}")

            # Example batch generation
            requests = [
                InferenceRequest(
                    question="What is the capital of France?",
                    table=Table(columns=["Country", "Capital"], rows=[["France", "Paris"]]),
                    ground_truth_answers=["Paris"],
                ),
                InferenceRequest(
                    question="What is machine learning?",
                    table=Table(
                        columns=["Term", "Definition"],
                        rows=[["ML", "Machine Learning"]],
                    ),
                    ground_truth_answers=["Machine Learning"],
                ),
            ]

            print("Generating responses with logging...")
            results = server.generate_batch(requests)

            for i, result in enumerate(results):
                print(f"Request {i + 1}: {result}")

    finally:
        server.shutdown()  # This will also generate summary report
