from leap.core import ExecutionMetrics
from leap.evaluation.metrics import check_denotation, to_value_list


def find_matching_answers(target_values, predicted_values):
    """Find which target answers have matches in predicted values"""
    matched = []
    for target in target_values:
        for predicted in predicted_values:
            if target.match(predicted):
                # Get the normalized string representation
                matched.append(target.normalized)
                break
    return matched


def calculate_execution_accuracy_with_dataset_answers(
    action_history, final_table, ground_truth_answers, original_table, generated_answers=None
) -> ExecutionMetrics:
    """Calculate execution accuracy using WikiTableQuestions evaluator logic

    Args:
        action_history: List of action strings
        final_table: Final table after executing actions
        ground_truth_answers: List of expected answers
        original_table: Original table before any actions
        generated_answers: Optional list of LLM-generated answers from Query(T,Q)

    Returns:
        ExecutionMetrics object with all evaluation results
    """
    # Initialize values
    execution_accuracy = 0.0
    terminated_properly = False
    answer_found_in_final = False
    answer_found_in_original = False
    matched_answers_final = []
    matched_answers_original = []
    execution_error = None
    final_table_size = None

    try:
        # Check if sequence terminated properly
        # With Query(T,Q), action history ends with "end()" then "direct_query()"
        # So we need to check if "end()" appears in the sequence, not just as last action
        has_end = any(action.startswith("end") for action in action_history)
        has_direct_query = "direct_query()" in action_history

        if has_direct_query:
            # For Query(T,Q) workflow: must have end() before direct_query()
            terminated_properly = (
                len(action_history) > 1  # Must have more than just one action
                and has_end  # Must have end() somewhere
                and not all(action.startswith("end") for action in action_history)  # Not all actions are end
            )
        else:
            # Original logic: last action must be end()
            terminated_properly = (
                len(action_history) > 1  # Must have more than just one action
                and action_history[-1].startswith("end")  # Last action must be end
                and not all(action.startswith("end") for action in action_history[:-1])  # Not all prior actions are end
            )

        print(f"[EVAL] action_history: {action_history}")
        print(f"[EVAL] has_end: {has_end}, has_direct_query: {has_direct_query}, terminated_properly: {terminated_properly}")

        # Convert ground truth answers to Value objects using evaluator logic
        target_values = to_value_list(ground_truth_answers)

        # Extract and evaluate original table
        original_table_values = original_table.extract_values()
        original_predicted_values = to_value_list(original_table_values)

        # Check if original table contains the answer using evaluator logic
        answer_found_in_original = check_denotation(target_values, original_predicted_values)
        if answer_found_in_original:
            # Find which answers matched in original table
            matched_answers_original = find_matching_answers(target_values, original_predicted_values)

        # Check final table OR generated answers
        if final_table:
            final_table_size = final_table.get_size()

            # If we have generated answers from Query(T,Q), use those instead of table values
            if generated_answers is not None and "direct_query()" in action_history:
                print(f"[EVAL] Using generated answers for evaluation: {generated_answers}")
                # Use LLM-generated answers for evaluation
                final_predicted_values = to_value_list(generated_answers)

                # Check if generated answers match ground truth
                answer_found_in_final = check_denotation(target_values, final_predicted_values)
                if answer_found_in_final:
                    matched_answers_final = find_matching_answers(target_values, final_predicted_values)
                print(f"[EVAL] answer_found_in_final: {answer_found_in_final}, matched: {matched_answers_final}")
            else:
                print(
                    f"[EVAL] Using table-based evaluation (generated_answers={generated_answers},"
                    "has_direct_query={'direct_query()' in action_history})"
                )
                # Fall back to table-based evaluation
                final_table_values = final_table.extract_values()
                final_predicted_values = to_value_list(final_table_values)

                # Check if final table contains the answer using evaluator logic
                answer_found_in_final = check_denotation(target_values, final_predicted_values)
                if answer_found_in_final:
                    # Find which answers matched in final table
                    matched_answers_final = find_matching_answers(target_values, final_predicted_values)

            # Calculate execution accuracy
            print(f"[EVAL] terminated_properly: {terminated_properly}, answer_found_in_final: {answer_found_in_final}")
            if terminated_properly and answer_found_in_final:
                execution_accuracy = 1.0
            else:
                execution_accuracy = 0.0
            print(f"[EVAL] execution_accuracy: {execution_accuracy}")
        else:
            execution_error = "No final table produced"

    except Exception as e:
        execution_error = str(e)

    return ExecutionMetrics(
        execution_accuracy=execution_accuracy,
        answer_found_in_final=answer_found_in_final,
        answer_found_in_original=answer_found_in_original,
        terminated_properly=terminated_properly,
        matched_answers_final=matched_answers_final,
        matched_answers_original=matched_answers_original,
        num_actions=len(action_history),
        final_table_size=final_table_size,
        execution_error=execution_error,
        evaluation_method="wikitablequestions_logic",
    )
