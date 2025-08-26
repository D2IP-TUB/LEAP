"""
vLLM Server Module with Integrated Logging Support

This module contains the worker process and parallel engine classes for running
vLLM inference across multiple GPUs with comprehensive table logging support.
"""

import os
import time
import uuid
import asyncio
import multiprocessing as mp
import queue
from typing import List, Dict, Any, Optional, Tuple, Callable
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# Import the table logger
from table_logger import TableLogger


class VLLMWorkerProcess(mp.Process):
    """Individual worker process for vLLM inference with logging support"""
    
    def __init__(self, 
                 worker_id: int, 
                 gpu_ids: List[int], 
                 model_id: str,
                 input_queue: mp.Queue, 
                 output_queue: mp.Queue,
                 generation_config: Dict[str, Any] = None):
        """
        Initialize vLLM worker process
        
        Args:
            worker_id: Unique identifier for this worker
            gpu_ids: List of GPU IDs to use for this worker
            model_id: HuggingFace model identifier
            input_queue: Queue for receiving inference requests
            output_queue: Queue for sending results back
            generation_config: Configuration for generation behavior and logging
        """
        super().__init__()
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids
        self.model_id = model_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.generation_config = generation_config or {}
        
        # Extract generation config
        self.use_constraints = self.generation_config.get('use_constraints', True)
        self.use_cot = self.generation_config.get('use_cot', False)
        self.constraint_processors = self.generation_config.get('constraint_processors', {})
        self.generation_functions = self.generation_config.get('generation_functions', {})
        
        # Logging configuration - use shared logger
        self.logging_config = self.generation_config.get('logging_config', {})
        self.logger = None  # Will use shared logger from main process
        
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
            self.output_queue.put(('ready', self.worker_id, None))
            
            # Process requests
            asyncio.run(self._process_requests())
            
        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
            self.output_queue.put(('error', self.worker_id, str(e)))
    
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
        
        # Get engine configuration from generation config
        engine_config = self.generation_config.get('engine_config', {})
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            trust_remote_code=engine_config.get('trust_remote_code', True),
            max_model_len=engine_config.get('max_model_len', 1024),
            gpu_memory_utilization=engine_config.get('gpu_memory_utilization', 0.8),
            tensor_parallel_size=engine_config.get('tensor_parallel_size', 1),
            max_num_batched_tokens=engine_config.get('max_num_batched_tokens', 8192),
            max_num_seqs=engine_config.get('max_num_seqs', 32),
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
                try:
                    result = await self._process_single_request(request, state_machines)
                    
                    # Send result back
                    self.output_queue.put(('result', request['request_id'], result))
                    
                except Exception as e:
                    print(f"Worker {self.worker_id} error processing request: {e}")
                    self.output_queue.put(('error', request['request_id'], str(e)))
                
                # Clean up state machines for this request
                request_id = request['request_id']
                to_delete = [key for key in state_machines if key.startswith(request_id)]
                for key in to_delete:
                    del state_machines[key]
                    
            except Exception as e:
                print(f"Worker {self.worker_id} error in main loop: {e}")
    
    async def _process_single_request(self, request: Dict[str, Any], state_machines: Dict) -> Dict[str, Any]:
        """
        Process a single request using the configured generation function with logging
        
        Args:
            request: Request dictionary containing all necessary data
            state_machines: Shared state machines for constraint processing
            
        Returns:
            Result dictionary
        """
        request_id = request['request_id']
        generation_mode = self._get_generation_mode_string()
        
        # Get the appropriate generation function
        if self.use_cot:
            generation_func = self.generation_functions.get('cot_generation')
        else:
            generation_func = self.generation_functions.get('iterative_generation')
        
        if not generation_func:
            raise ValueError(f"No generation function configured for mode: {generation_mode}")
        
        # Call the generation function with worker context and logging
        # Instead of using separate loggers, send logging data back to main process
        return await generation_func(
            request=request,
            worker=self,
            state_machines=state_machines,
            logging_callback=self._send_log_entry  # Send logs back to main process
        )
    
    def _send_log_entry(self, request_id: str, step: int, action: str, table: Dict[str, Any],
                       success: bool = True, failure_type: str = None, generation_mode: str = None):
        """Send logging data back to main process via output queue"""
        if self.logging_config.get('enable_logging', False):
            log_data = {
                'request_id': request_id,
                'step': step, 
                'action': action,
                'table': table,
                'success': success,
                'failure_type': failure_type,
                'generation_mode': generation_mode,
                'worker_id': self.worker_id
            }
            # Send log entry back to main process
            self.output_queue.put(('log_entry', request_id, log_data))
    
    async def generate_text(self, 
                          prompt: str, 
                          request_id: str, 
                          sampling_params: SamplingParams) -> str:
        """
        Generate text using the vLLM engine
        
        Args:
            prompt: Input prompt
            request_id: Unique request identifier
            sampling_params: Sampling parameters for generation
            
        Returns:
            Generated text
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
                
        except Exception as e:
            print(f"Generation error in worker {self.worker_id}: {e}")
            return ""


class ProcessParallelVLLM:
    """Process-based parallel vLLM engine with integrated logging"""
    
    def __init__(self, 
                 model_id: str, 
                 num_workers: int = 4,
                 gpu_allocation: List[int] = None,
                 generation_config: Dict[str, Any] = None):
        """
        Initialize parallel vLLM engine
        
        Args:
            model_id: HuggingFace model identifier
            num_workers: Number of worker processes
            gpu_allocation: List of GPU IDs to use (auto-detected if None)
            generation_config: Configuration for generation behavior and logging
        """
        self.model_id = model_id
        self.num_workers = num_workers
        self.generation_config = generation_config or {}
        
        # GPU allocation
        if gpu_allocation is None:
            # Default GPU allocation - adjust based on your system
            self.available_gpus = [1, 2, 3, 4, 5, 6, 7]
        else:
            self.available_gpus = gpu_allocation
        
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
        logging_config = self.generation_config.get('logging_config', {})
        if logging_config.get('enable_logging', False):
            log_dir = logging_config.get('log_dir', 'table_logs')
            
            self.main_logger = TableLogger(
                log_dir=log_dir,
                enable_logging=True,
                save_readable_tables=logging_config.get('save_readable_tables', True),
                compress_logs=logging_config.get('compress_logs', False),
                log_format=logging_config.get('log_format', 'readable'),
                max_table_chars=logging_config.get('max_table_chars', 10000)
            )
            print(f"Main logger initialized: {log_dir}")
    
    def _create_workers(self):
        """Create worker processes with GPU allocation"""

        tensor_parallel_size = self.generation_config["engine_config"]["tensor_parallel_size"]

        if tensor_parallel_size > 1:

            print(f"Spawning workers with {tensor_parallel_size} GPUs allocated.")

        for worker_id in range(self.num_workers):
            if tensor_parallel_size > 1:
                first_gpu = worker_id * tensor_parallel_size
                final_gpu = first_gpu + tensor_parallel_size
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
                generation_config=self.generation_config
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
                if msg_type == 'ready':
                    print(f"Worker {worker_id} is ready")
                    ready_count += 1
                elif msg_type == 'error':
                    print(f"Worker {worker_id} failed: {data}")
            except queue.Empty:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    print(f"Timeout waiting for workers to load models ({timeout}s)")
                    break
        
        self._workers_ready = (ready_count == self.num_workers)
        
        if self._workers_ready:
            print(f"All {self.num_workers} workers are ready")
        else:
            print(f"Only {ready_count}/{self.num_workers} workers ready")
        
        return self._workers_ready
    
    def _get_generation_mode_description(self) -> str:
        """Get description of current generation mode"""
        use_cot = self.generation_config.get('use_cot', False)
        use_constraints = self.generation_config.get('use_constraints', True)
        
        if use_cot:
            return "Chain-of-Table (dynamic_plan + generate_args)"
        elif use_constraints:
            return "Constrained generation"
        else:
            return "Unconstrained generation with post-processing"
    
    def generate_batch(self, 
                      requests: List[Dict[str, Any]], 
                      timeout_per_request: int = 100) -> List[Dict[str, Any]]:
        """
        Generate responses for batch of requests
        
        Args:
            requests: List of request dictionaries
            timeout_per_request: Timeout per request in seconds
            
        Returns:
            List of result dictionaries in same order as requests
        """
        if not self._workers_ready:
            raise RuntimeError("Workers not ready. Call start_workers() first.")
        
        # Send all requests to queue with unique IDs
        request_ids = []
        for i, request in enumerate(requests):
            request_id = f"req_{i}_{uuid.uuid4().hex[:8]}"
            request_ids.append(request_id)
            
            # Add request_id to the request
            request_with_id = request.copy()
            request_with_id['request_id'] = request_id
            
            self.input_queue.put(request_with_id)
        
        # Collect results
        results = {}
        completed = 0
        total_requests = len(request_ids)
        
        while completed < total_requests:
            try:
                msg_type, req_id, data = self.output_queue.get(timeout=timeout_per_request)
                if msg_type == 'result':
                    results[req_id] = data
                    completed += 1
                    print(f"Completed {completed}/{total_requests} requests")
                elif msg_type == 'log_entry':
                    # Handle log entries from workers
                    if self.main_logger:
                        log_data = data
                        self.main_logger.log_table_state(
                            request_id=log_data['request_id'],
                            step=log_data['step'],
                            action=log_data['action'],
                            table=log_data['table'],
                            success=log_data['success'],
                            failure_type=log_data['failure_type'],
                            generation_mode=log_data['generation_mode']
                        )
                elif msg_type == 'error':
                    print(f"Error for request {req_id}: {data}")
                    # Store error result
                    results[req_id] = {'error': data}
                    completed += 1
            except queue.Empty:
                print(f"Timeout waiting for results (completed {completed}/{total_requests})")
                break
        
        # Return results in original order, with default empty results for missing ones
        return [results.get(req_id, {'error': 'Request not completed'}) for req_id in request_ids]
    
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
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration"""
        return self.generation_config.copy()
    
    def get_logging_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        if self.main_logger:
            return self.main_logger.get_logging_stats()
        else:
            return {"enabled": False}


def create_generation_config(use_constraints: bool = True,
                           use_cot: bool = False,
                           use_global_constraints: bool = True,  # NEW parameter
                           constraint_processors: Dict[str, Callable] = None,
                           generation_functions: Dict[str, Callable] = None,
                           engine_config: Dict[str, Any] = None,
                           logging_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a generation configuration dictionary with logging support
    
    Args:
        use_constraints: Whether to use constrained generation
        use_cot: Whether to use Chain-of-Table approach
        use_global_constraints: Whether to apply global action constraints (no duplicates, end rules)
        constraint_processors: Dictionary of constraint processor functions
        generation_functions: Dictionary of generation functions
        engine_config: vLLM engine configuration
        logging_config: Logging configuration
        
    Returns:
        Configuration dictionary
    """
    config = {
        'use_constraints': use_constraints,
        'use_cot': use_cot,
        'use_global_constraints': use_global_constraints,  # NEW field
        'constraint_processors': constraint_processors or {},
        'generation_functions': generation_functions or {},
        'engine_config': engine_config or {
            'trust_remote_code': True,
            'max_model_len': 1024,
            'gpu_memory_utilization': 0.8,
            'tensor_parallel_size': 4, # tensor parralell -> weights split between n GPUs
            'max_num_batched_tokens': 8192,
            'max_num_seqs': 32,
        },
        'logging_config': logging_config or {
            'enable_logging': True,
            'log_dir': 'table_logs',
            'save_readable_tables': True,
            'compress_logs': False,
            'log_format': 'readable',
            'max_table_chars': 10000
        }
    }
    
    return config


def create_logging_config(enable_logging: bool = True,
                         log_dir: str = "table_logs",
                         save_readable_tables: bool = True,
                         compress_logs: bool = False,
                         log_format: str = "readable",
                         max_table_chars: int = 10000) -> Dict[str, Any]:
    """
    Create a logging configuration dictionary
    
    Args:
        enable_logging: Whether to enable logging
        log_dir: Directory for log files
        save_readable_tables: Whether to save full tables as CSV
        compress_logs: Whether to compress log files
        log_format: Format for logs ("json", "csv", "readable", "pickle")
        max_table_chars: Maximum characters for table serialization
        
    Returns:
        Logging configuration dictionary
    """
    return {
        'enable_logging': enable_logging,
        'log_dir': log_dir,
        'save_readable_tables': save_readable_tables,
        'compress_logs': compress_logs,
        'log_format': log_format,
        'max_table_chars': max_table_chars
    }


# Example usage and configuration helpers
def setup_standard_vllm_server(model_id: str,
                              num_workers: int = 4,
                              gpu_allocation: List[int] = None,
                              use_constraints: bool = True,
                              use_cot: bool = False,
                              enable_logging: bool = True,
                              log_dir: str = "table_logs") -> ProcessParallelVLLM:
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
        
    Returns:
        Configured ProcessParallelVLLM instance
    """
    logging_config = create_logging_config(
        enable_logging=enable_logging,
        log_dir=log_dir,
        save_readable_tables=True,
        log_format="readable"
    )
    
    generation_config = create_generation_config(
        use_constraints=use_constraints,
        use_cot=use_cot,
        logging_config=logging_config
    )
    
    return ProcessParallelVLLM(
        model_id=model_id,
        num_workers=num_workers,
        gpu_allocation=gpu_allocation,
        generation_config=generation_config
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
        log_dir="example_logs"
    )
    
    try:
        # Start workers
        if server.start_workers():
            print("Server started successfully!")
            print(f"Logging stats: {server.get_logging_stats()}")
            
            # Example batch generation
            requests = [
                {
                    "question": "What is the capital of France?",
                    "table": {"columns": ["Country", "Capital"], "rows": [["France", "Paris"]]},
                    "ground_truth_answers": ["Paris"]
                },
                {
                    "question": "What is machine learning?", 
                    "table": {"columns": ["Term", "Definition"], "rows": [["ML", "Machine Learning"]]},
                    "ground_truth_answers": ["Machine Learning"]
                }
            ]
            
            print("Generating responses with logging...")
            results = server.generate_batch(requests)
            
            for i, result in enumerate(results):
                print(f"Request {i+1}: {result}")
        
    finally:
        server.shutdown()  # This will also generate summary report