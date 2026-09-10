from leap.utils.profiler import AggregateProfiler, RequestProfiler


def test_aggregate_profiler_reset_isolates_runs():
    profiler = AggregateProfiler()
    profiler.add_request_profile(2.0, {"generation": 1.5}, 3, {"mcp_startup": 0.2})

    profiler.reset()

    assert profiler.request_times == []
    assert profiler.step_counts == []
    assert dict(profiler.operation_times) == {}
    assert dict(profiler.diagnostic_times) == {}


def test_request_profiler_keeps_diagnostics_out_of_operation_totals():
    profiler = RequestProfiler("request-1")

    profiler.record_timing("mcp_table_transformation", 0.3)
    profiler.record_timing("mcp_startup", 0.2, diagnostic=True)
    profiler.record_timing("mcp_call", 0.1, diagnostic=True)

    assert profiler.timings == {"mcp_table_transformation": 0.3}
    assert profiler.diagnostic_timings == {"mcp_startup": 0.2, "mcp_call": 0.1}
