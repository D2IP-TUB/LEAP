from leap.utils.profiler import AggregateProfiler


def test_aggregate_profiler_reset_isolates_runs():
    profiler = AggregateProfiler()
    profiler.add_request_profile(2.0, {"generation": 1.5}, 3)

    profiler.reset()

    assert profiler.request_times == []
    assert profiler.step_counts == []
    assert dict(profiler.operation_times) == {}
