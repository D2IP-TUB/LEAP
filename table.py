from typing import List, Tuple, Dict, Any, Optional

# Utility functions (table manipulation, parsing, etc.)
def serialize_table_to_csv(table, max_chars=1500):
    """Convert table to CSV string with limited characters"""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(table['columns'])
    
    char_count = len(','.join(table['columns']))
    for i, row in enumerate(table['rows']):
        if i >= 10 or char_count > max_chars:
            break
        row_str = ','.join(str(cell) for cell in row)
        if char_count + len(row_str) > max_chars:
            break
        writer.writerow(row)
        char_count += len(row_str)
    return output.getvalue().strip()

def apply_action(table: Dict[str, Any], action: str, args: List) -> Optional[Dict[str, Any]]:
    """Apply action to table and return new table state"""
    if action == "select_row":
        # Validate row indices
        valid_indices = []
        for idx in args:
            if isinstance(idx, int) and 0 <= idx < len(table['rows']):
                valid_indices.append(idx)
            elif isinstance(idx, str) and idx.isdigit():
                idx_int = int(idx)
                if 0 <= idx_int < len(table['rows']):
                    valid_indices.append(idx_int)
        
        if not valid_indices:
            return None
            
        # Create new table with selected rows
        new_rows = [table['rows'][i] for i in valid_indices]
        return {'columns': table['columns'], 'rows': new_rows}
    
    elif action == "select_column":
        valid_columns = [col for col in args if col in table['columns']]
        
        if not valid_columns:
            return None
            
        # Create new table with selected columns
        col_indices = [table['columns'].index(col) for col in valid_columns]
        new_rows = []
        for row in table['rows']:
            new_rows.append([row[i] for i in col_indices])
        
        return {'columns': valid_columns, 'rows': new_rows}
    
    return None

def extract_table_values_for_eval(table):
    """Extract all values from a table as a flat list of strings for evaluation"""
    if not table or not table.get('rows'):
        return []
    
    values = []
    for row in table['rows']:
        for cell in row:
            if cell is not None and str(cell).strip():
                values.append(str(cell).strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    
    return unique_values

