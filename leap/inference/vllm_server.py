"""
vLLM Server Module with Integrated Logging Support

This module contains the worker process and parallel engine classes for running
vLLM inference across multiple GPUs with comprehensive table logging support.
"""

import asyncio
import multiprocessing as mp
import os
import queue
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from leap.config.loader import GenerationConfig, LoggingConfig, TokenizerConfig
from leap.core import ExecutionMetrics, InferenceRequest, InferenceResult, Table

# Import the table logger
from leap.utils.table_logger import TableLogger


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

        # Extract commonly used fields for convenience
        self.use_constraints = generation_config.use_constraints
        self.use_cot = generation_config.use_chain_of_table
        self.use_global_constraints = generation_config.use_global_constraints

        # vLLM components
        self.engine = None
        self.tokenizer = None
        self.max_model_len = None
        self.model_loaded = mp.Event()

    def run(self):
        """Main worker process loop"""
        try:
            # Set GPU visibility for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

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

    def _get_generation_mode_string(self) -> str:
        """Get descriptive string for generation mode"""
        if self.use_cot:
            constraint_desc = "with_constraints" if self.use_constraints else "without_constraints"
            return f"chain_of_table_{constraint_desc}"
        elif self.use_constraints:
            return "constrained"
        else:
            return "unconstrained_with_postprocessing"

    def _init_engine(self):
        """Initialize the vLLM engine"""
        print(f"Worker {self.worker_id}: Starting model loading...")
        start_time = time.time()

        # Use typed config instead of dict
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_batched_tokens=8192,
            max_num_seqs=32,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        tokenizer_group = self.engine.engine.tokenizer
        self.tokenizer = tokenizer_group.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = self.engine.engine.model_config.max_model_len

        load_time = time.time() - start_time
        print(f"Worker {self.worker_id}: Model loaded in {load_time:.2f} seconds, max_model_len={self.max_model_len}")

    async def _process_requests(self):
        """Process incoming requests"""
        state_machines = {}

        while True:
            try:
                # Get request from queue
                try:
                    request = self.input_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if request is None:  # Shutdown signal
                    break

                # Process the request using configured generation function
                # Error handling is done inside _process_single_request
                result = await self._process_single_request(request, state_machines)

                # Send result back (result may contain error info)
                self.output_queue.put(("result", request.request_id, result))

                # Clean up state machines for this request
                to_delete = [key for key in state_machines if key.startswith(request.request_id)]
                for key in to_delete:
                    del state_machines[key]

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
            generation_mode = self._get_generation_mode_string()

            # Get the appropriate generation function
            if self.use_cot:
                generation_func = self.generation_functions.get("cot_generation")
            else:
                generation_func = self.generation_functions.get("iterative_generation")

            if not generation_func:
                raise ValueError(f"No generation function configured for mode: {generation_mode}")

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
        """
        self.model_id = model_id
        self.num_workers = num_workers

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
        use_cot = self.generation_config.use_chain_of_table
        use_constraints = self.generation_config.use_constraints

        if use_cot:
            return "Chain-of-Table (dynamic_plan + generate_args)"
        elif use_constraints:
            return "Constrained generation"
        else:
            return "Unconstrained generation with post-processing"

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

        # Send all requests to queue with unique IDs
        request_ids = []
        for i, request in enumerate(requests):
            request_id = f"req_{i}_{uuid.uuid4().hex[:8]}"
            request_ids.append(request_id)

            # Add request_id to the request using dataclass replace
            from dataclasses import replace

            request_with_id = replace(request, request_id=request_id)

            # Put the typed object directly in the queue
            self.input_queue.put(request_with_id)

        # Collect results
        results = {}
        completed = 0
        total_requests = len(request_ids)

        while completed < total_requests:
            try:
                msg_type, req_id, data = self.output_queue.get(timeout=timeout_per_request)
                if msg_type == "result":
                    # data is already an InferenceResult object
                    results[req_id] = data
                    completed += 1
                    print(f"Completed {completed}/{total_requests} requests")
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
                    print(f"Error for request {req_id}: {data}")
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
            except queue.Empty:
                print(f"Timeout waiting for results (completed {completed}/{total_requests})")
                break

        # Return results in original order, with default error results for missing ones
        default_error = InferenceResult(
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
            request_id="",
            question="",
            ground_truth_answers=[],
        )
        return [results.get(req_id, default_error) for req_id in request_ids]

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

    def shutdown(self):
        """Shutdown all worker processes"""
        print("Shutting down workers...")

        # Send shutdown signals
        for _ in self.workers:
            self.input_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Force terminating worker {worker.worker_id}")
                worker.terminate()
                worker.join()

        self._workers_ready = False

        # Final summary report if logging enabled
        if self.main_logger:
            print("\nGenerating final summary report...")
            generation_mode = self._get_generation_mode_description()
            self.write_summary_report(generation_mode)

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
    generation_config = replace(
        app_config.generation,
        use_constraints=use_constraints,
        use_chain_of_table=use_cot,
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
