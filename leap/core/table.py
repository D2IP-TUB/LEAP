"""Table abstraction - single source of truth for table operations"""

import csv
import io
from dataclasses import dataclass
from typing import List, Any, Optional, Dict


@dataclass(frozen=True)
class Table:
    """Immutable table representation with operations"""

    columns: tuple[str, ...]  # Immutable tuple
    rows: tuple[tuple[Any, ...], ...]  # Immutable nested tuples

    def __init__(self, columns: List[str], rows: List[List[Any]]):
        """Initialize with mutable lists, convert to immutable internally"""
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(self, "columns", tuple(columns))
        object.__setattr__(self, "rows", tuple(tuple(row) for row in rows))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Table":
        """Create Table from dictionary format (WikiTableQuestions format)"""
        return cls(
            columns=data.get("columns", data.get("header", [])),
            rows=data.get("rows", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization"""
        return {"columns": list(self.columns), "rows": [list(row) for row in self.rows]}

    def to_csv(
        self, max_chars: int = 1500, max_rows: int = 10, crop: bool = True
    ) -> str:
        """
        Serialize table to CSV string with budget constraints.

        Single source of truth for CSV serialization - eliminates duplication
        between table.py and table_logger.py
        """
        out = io.StringIO()
        writer = csv.writer(out)

        # Write header
        writer.writerow([" "] + list(self.columns))
        current = out.getvalue()

        if crop and len(current) > max_chars:
            return current[:max_chars].rstrip()

        rows_written = 0
        for i, row in enumerate(self.rows):
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

    def select_rows(self, indices: List[int]) -> Optional["Table"]:
        """
        Select specific rows by index.

        Returns None if no valid indices provided.
        """
        valid_indices = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(self.rows):
                valid_indices.append(idx)
            elif isinstance(idx, str) and idx.isdigit():
                idx_int = int(idx)
                if 0 <= idx_int < len(self.rows):
                    valid_indices.append(idx_int)

        if not valid_indices:
            return None

        new_rows = [self.rows[i] for i in valid_indices]
        return Table(columns=list(self.columns), rows=new_rows)

    def select_columns(self, column_names: List[str]) -> Optional["Table"]:
        """
        Select specific columns by name.

        Returns None if no valid columns provided.
        """
        valid_columns = [col for col in column_names if col in self.columns]

        if not valid_columns:
            return None

        col_indices = [self.columns.index(col) for col in valid_columns]
        new_rows = []
        for row in self.rows:
            new_rows.append([row[i] for i in col_indices])

        return Table(columns=valid_columns, rows=new_rows)

    def extract_values(self) -> List[str]:
        """
        Extract all values from table as flat list of unique strings.

        Used for evaluation/answer matching.
        """
        if not self.rows:
            return []

        values = []
        for row in self.rows:
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

    def get_summary(self) -> str:
        """Get dimension summary for logging"""
        return f"{len(self.rows)} rows × {len(self.columns)} columns"

    def get_size(self) -> tuple[int, int]:
        """Get (num_rows, num_columns) tuple"""
        return (len(self.rows), len(self.columns))

    def __repr__(self) -> str:
        return f"Table(columns={len(self.columns)}, rows={len(self.rows)})"
