"""Tests for error handling in the inference pipeline"""

from leap.core import ExecutionMetrics, InferenceRequest, InferenceResult, Table


def test_inference_result_from_dict_with_error():
    """Test that InferenceResult.from_dict handles error dictionaries correctly"""
    error_dict = {
        "error": "The decoder prompt (length 1055) is longer than the maximum model length of 1024.",
        "request_id": "req_test_123",
        "question": "What is the capital of France?",
        "ground_truth_answers": ["Paris"],
    }

    result = InferenceResult.from_dict(error_dict)

    # Verify the result was created with default values
    assert result.request_id == "req_test_123"
    assert result.question == "What is the capital of France?"
    assert result.ground_truth_answers == ["Paris"]
    assert result.action_history == []
    assert result.final_table.columns == ()  # Table uses tuples
    assert result.final_table.rows == ()  # Table uses tuples

    # Verify the error metrics
    assert result.execution_metrics.execution_accuracy == 0.0
    assert result.execution_metrics.answer_found_in_final is False
    assert result.execution_metrics.answer_found_in_original is False
    assert result.execution_metrics.terminated_properly is False
    assert result.execution_metrics.num_actions == 0
    assert "maximum model length" in result.execution_metrics.execution_error


def test_inference_result_from_dict_with_normal_result():
    """Test that InferenceResult.from_dict handles normal results correctly"""
    normal_dict = {
        "action_history": ["select_row([0, 1])", "end()"],
        "final_table": {
            "columns": ["Country", "Capital"],
            "rows": [["France", "Paris"], ["Germany", "Berlin"]],
        },
        "execution_accuracy_metrics": {
            "execution_accuracy": 1.0,
            "answer_found_in_final": True,
            "answer_found_in_original": True,
            "terminated_properly": True,
            "matched_answers_final": ["Paris"],
            "matched_answers_original": ["Paris"],
            "num_actions": 2,
        },
        "request_id": "req_test_456",
        "question": "What is the capital of France?",
        "ground_truth_answers": ["Paris"],
    }

    result = InferenceResult.from_dict(normal_dict)

    # Verify the result was created correctly
    assert result.request_id == "req_test_456"
    assert result.question == "What is the capital of France?"
    assert result.ground_truth_answers == ["Paris"]
    assert result.action_history == ["select_row([0, 1])", "end()"]
    assert len(result.final_table.columns) == 2
    assert len(result.final_table.rows) == 2
    assert result.execution_metrics.execution_accuracy == 1.0
    assert result.execution_metrics.answer_found_in_final is True


def test_inference_result_to_dict():
    """Test that InferenceResult.to_dict produces the correct dictionary format"""
    result = InferenceResult(
        action_history=["select_row([0])"],
        final_table=Table(columns=["A"], rows=[["1"]]),
        execution_metrics=ExecutionMetrics(
            execution_accuracy=0.5,
            answer_found_in_final=True,
            answer_found_in_original=False,
            terminated_properly=True,
            matched_answers_final=["1"],
            matched_answers_original=[],
            num_actions=1,
        ),
        request_id="req_789",
        question="Test question",
        ground_truth_answers=["1"],
        generated_answers=["1"],
    )

    result_dict = result.to_dict()

    assert result_dict["request_id"] == "req_789"
    assert result_dict["question"] == "Test question"
    assert result_dict["ground_truth_answers"] == ["1"]
    assert result_dict["generated_answers"] == ["1"]
    assert result_dict["action_history"] == ["select_row([0])"]
    assert "execution_accuracy_metrics" in result_dict
    assert result_dict["execution_accuracy_metrics"]["execution_accuracy"] == 0.5


def test_inference_request_from_example_uses_stable_example_id():
    example = {
        "question": "What is the capital of France?",
        "table": {"header": ["Country", "Capital"], "rows": [["France", "Paris"]]},
        "answers": ["Paris"],
    }

    request = InferenceRequest.from_example(example, index=7)

    assert request.request_id == "example_7"


def test_inference_result_round_trip_with_error():
    """Test that error results can survive a to_dict -> from_dict round trip"""
    # Create an error result
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
            execution_error="Test error message",
        ),
        request_id="req_error",
        question="Error question",
        ground_truth_answers=["answer"],
    )

    # Round trip
    result_dict = error_result.to_dict()
    recovered_result = InferenceResult.from_dict(result_dict)

    # Verify
    assert recovered_result.request_id == "req_error"
    assert recovered_result.execution_metrics.execution_error == "Test error message"
    assert recovered_result.execution_metrics.execution_accuracy == 0.0
