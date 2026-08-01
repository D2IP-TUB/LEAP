"""Performance profiling utilities for LEAP"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List


class RequestProfiler:
    """Per-request profiler for timing operations within a single request"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.timings: Dict[str, float] = {}
        self.start_time = time.perf_counter()

    @contextmanager
    def time_operation(self, operation: str):
        """Time a specific operation within the request"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings[operation] = self.timings.get(operation, 0.0) + duration

    def get_total_time(self) -> float:
        """Get total request time since profiler creation"""
        return time.perf_counter() - self.start_time


class AggregateProfiler:
    """Collects profiling data from multiple requests and generates summary statistics"""

    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.request_times: List[float] = []
        self.step_counts: List[int] = []

    def add_request_profile(self, total_time: float, operation_timings: Dict[str, float], num_steps: int):
        """Add profiling data from a completed request"""
        self.request_times.append(total_time)
        self.step_counts.append(num_steps)
        for operation, time_spent in operation_timings.items():
            self.operation_times[operation].append(time_spent)

    def print_summary(self):
        """Print comprehensive summary of all profiling data"""
        if not self.request_times:
            print("No profiling data collected")
            return

        num_requests = len(self.request_times)

        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 80)
        print("OPERATION BREAKDOWN (per request average):")
        print("-" * 80)

        operation_totals = {}
        for operation, times in self.operation_times.items():
            operation_totals[operation] = sum(times) / num_requests

        total_accounted = sum(operation_totals.values())
        sorted_ops = sorted(operation_totals.items(), key=lambda x: x[1], reverse=True)

        for operation, avg_time_spent in sorted_ops:
            percentage = (avg_time_spent / total_accounted) * 100 if total_accounted > 0 else 0
            times = self.operation_times[operation]
            min_time = min(times)
            max_time = max(times)
            print(f"  {operation:30s}: {avg_time_spent:6.3f}s ({percentage:5.1f}%) [min: {min_time:.3f}s, max: {max_time:.3f}s]")

        print("\n" + "=" * 80)


# Global instance
_global_aggregate_profiler = AggregateProfiler()


def get_aggregate_profiler() -> AggregateProfiler:
    """Get the global aggregate profiler instance"""
    return _global_aggregate_profiler
