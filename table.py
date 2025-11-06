import csv
import io
from typing import List, Dict, Any, Optional


def serialize_table_to_csv(
    table: Dict[str, Any],
    max_chars = 1500,
    max_rows = 10,
    crop = True
) -> str:
    out = io.StringIO()
    writer = csv.writer(out)

    # Write header
    writer.writerow([" "] + list(table["columns"]))
    current = out.getvalue()

    if crop and len(current) > max_chars:
            return current[:max_chars].rstrip()
    
    rows_written = 0
    for i, row in enumerate(table["rows"]):
        if crop and rows_written >= max_rows:
            break

        # Measure the exact CSV for this row using a temp buffer
        temp = io.StringIO()
        temp_writer = csv.writer(temp)
        temp_writer.writerow([f"row {i}"] + list(row))
        delta = temp.getvalue()

        # Check character budget if provided
        if crop and len(current) + len(delta) > max_chars:
            break

        # Commit the row
        out.write(delta)
        current += delta
        rows_written += 1

    return current.rstrip()

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

