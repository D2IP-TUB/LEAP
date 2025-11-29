from leap.evaluation.metrics import to_value_list, check_denotation
from leap.core import Table

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

def calculate_execution_accuracy_with_dataset_answers(action_history, final_table, ground_truth_answers, original_table):
    """Calculate execution accuracy using WikiTableQuestions evaluator logic"""
    result = {
        'execution_accuracy': 0.0,
        'terminated_properly': False,
        'answer_found_in_final': False,
        'answer_found_in_original': False,
        'final_table_values': [],
        'original_table_values': [],
        'matched_answers_final': [],
        'matched_answers_original': [],
        'execution_error': None,
        'num_actions': len(action_history),
        'final_table_size': None,
        'evaluation_method': 'wikitablequestions_logic'
    }
    
    try:
        # Check if sequence terminated properly
        result['terminated_properly'] = (
            len(action_history) > 1 and  # Must have more than just one action
            action_history[-1].startswith('end') and  # Last action must be end
            not all(action.startswith('end') for action in action_history[:-1])  # Not all prior actions are end
        )
        
        # Convert ground truth answers to Value objects using evaluator logic
        target_values = to_value_list(ground_truth_answers)

        # Extract and evaluate original table
        result['original_table_values'] = original_table.extract_values()
        original_predicted_values = to_value_list(result['original_table_values'])
        
        # Check if original table contains the answer using evaluator logic
        result['answer_found_in_original'] = check_denotation(target_values, original_predicted_values)
        if result['answer_found_in_original']:
            # Find which answers matched in original table
            result['matched_answers_original'] = find_matching_answers(target_values, original_predicted_values)
        
        # Check final table
        if final_table:
            result['final_table_size'] = final_table.get_size()
            result['final_table_values'] = final_table.extract_values()
            final_predicted_values = to_value_list(result['final_table_values'])
            
            # Check if final table contains the answer using evaluator logic
            result['answer_found_in_final'] = check_denotation(target_values, final_predicted_values)
            if result['answer_found_in_final']:
                # Find which answers matched in final table
                result['matched_answers_final'] = find_matching_answers(target_values, final_predicted_values)
            
            # Calculate execution accuracy
            if result['terminated_properly'] and result['answer_found_in_final']:
                result['execution_accuracy'] = 1.0
            else:
                result['execution_accuracy'] = 0.0
        else:
            result['execution_error'] = "No final table produced"
    
    except Exception as e:
        result['execution_error'] = str(e)
    
    return result
