"""
Table Logger Module - Extracted table logging functionality

This module handles all table transformation logging, analysis, and reporting
functionality that was previously embedded in the main processing logic.
"""

import json
import csv
import io
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LogEntry:
    """Structure for a single log entry"""

    request_id: str
    step: int
    action: str
    timestamp: float
    success: bool
    failure_type: Optional[str] = None
    generation_mode: Optional[str] = None
    model_type: Optional[str] = None
    table_summary: Optional[Dict[str, Any]] = None
    table_preview: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class TableLogger:
    """
    Handles table transformation logging with configurable output formats
    and comprehensive analysis capabilities.
    """

    def __init__(
        self,
        log_dir: str = "table_logs",
        enable_logging: bool = True,
        save_readable_tables: bool = True,
        compress_logs: bool = False,
        log_format: str = "readable",
        max_table_chars: int = 10000,
    ):
        """
        Initialize table logger

        Args:
            log_dir: Directory for log files
            enable_logging: Whether to enable logging
            save_readable_tables: Whether to save full tables as CSV
            compress_logs: Whether to compress log files
            log_format: Format for logs ("json", "csv", "readable", "pickle")
            max_table_chars: Maximum characters for table serialization
        """
        self.log_dir = Path(log_dir)
        self.enable_logging = enable_logging
        self.save_readable_tables = save_readable_tables
        self.compress_logs = compress_logs
        self.log_format = log_format
        self.max_table_chars = max_table_chars

        # Performance tracking
        self.log_entries: Dict[str, List[LogEntry]] = {}
        self.request_metadata: Dict[str, Dict[str, Any]] = {}

        if self.enable_logging:
            self.setup_logging_directory()

    def setup_logging_directory(self):
        """Create logging directory if it doesn't exist"""
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created table logging directory: {self.log_dir}")

    def serialize_table_to_csv(
        self, table: Dict[str, Any], max_chars: int = None
    ) -> str:
        """Convert table to CSV string with limited characters"""
        if max_chars is None:
            max_chars = self.max_table_chars
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(table["columns"])

        char_count = len(",".join(table["columns"]))
        for i, row in enumerate(table["rows"]):
            if i >= 10 or char_count > max_chars:
                break
            row_str = ",".join(str(cell) for cell in row)
            if char_count + len(row_str) > max_chars:
                break
            writer.writerow(row)
            char_count += len(row_str)
        return output.getvalue().strip()

    def get_table_summary(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Get a compact summary of table dimensions"""
        return {
            "num_rows": len(table["rows"]),
            "num_columns": len(table["columns"]),
            "columns": table["columns"][:5] + ["..."]
            if len(table["columns"]) > 5
            else table["columns"],
        }

    def log_table_state(
        self,
        request_id: str,
        step: int,
        action: str,
        table: Dict[str, Any],
        success: bool = True,
        failure_type: Optional[str] = None,
        generation_mode: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> None:
        """
        Log table state with comprehensive information

        Args:
            request_id: Unique request identifier
            step: Step number in the transformation sequence
            action: Action being performed
            table: Current table state
            success: Whether the action succeeded
            failure_type: Type of failure if unsuccessful
            generation_mode: Generation mode used
            model_type: LLM model used
        """
        if not self.enable_logging:
            return

        try:
            # Create log entry
            table_summary = self.get_table_summary(table)
            table_preview = None

            # Create readable table representation
            if self.save_readable_tables:
                table_csv = self.serialize_table_to_csv(
                    table, max_chars=self.max_table_chars
                )
                table_preview = table_csv.split("\n")[:6]  # First 5 rows + header

                # Save full table as separate CSV file
                self._save_table_csv(request_id, step, action, table)

            log_entry = LogEntry(
                request_id=request_id,
                step=step,
                action=action,
                timestamp=time.time(),
                success=success,
                failure_type=failure_type,
                generation_mode=generation_mode,
                model_type=model_type,
                table_summary=table_summary,
                table_preview=table_preview,
            )

            # Store in memory for analysis
            if request_id not in self.log_entries:
                self.log_entries[request_id] = []
            self.log_entries[request_id].append(log_entry)

            # Write to file
            self._write_log_entry(log_entry)

        except Exception as e:
            print(f"Warning: Failed to log table state: {e}")

    def _save_table_csv(
        self, request_id: str, step: int, action: str, table: Dict[str, Any]
    ) -> None:
        """Save full table as CSV file"""
        try:
            # Clean action name for filename
            clean_action = self._clean_filename(action)
            table_filename = f"{request_id}_step{step:02d}_{clean_action}.csv"
            table_path = self.log_dir / table_filename

            with open(table_path, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(table["columns"])
                for row in table["rows"]:
                    writer.writerow(row)
        except Exception as e:
            print(f"Warning: Failed to save table CSV: {e}")

    def _clean_filename(self, action: str) -> str:
        """Clean action string for use in filename"""
        clean_action = (
            action.replace("(", "_").replace(")", "").replace("[", "").replace("]", "")
        )
        clean_action = clean_action.replace('"', "").replace(",", "_").replace(" ", "_")
        return clean_action[:50]  # Limit length

    def _write_log_entry(self, log_entry: LogEntry) -> None:
        """Write log entry to file"""
        try:
            log_filename = f"{log_entry.request_id}_log.json"
            log_path = self.log_dir / log_filename

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(log_entry.to_dict(), indent=2) + "\n" + "-" * 80 + "\n"
                )
        except Exception as e:
            print(f"Warning: Failed to write log entry: {e}")

    def create_summary_report(self, generation_mode: str = None) -> Dict[str, Any]:
        """Create comprehensive summary report of all logged transformations"""
        if not self.enable_logging:
            return {"error": "Logging is disabled"}

        summary_data = {
            "total_requests": len(self.log_entries),
            "total_transformations": 0,
            "action_counts": {},
            "failure_type_counts": {},
            "generation_mode_counts": {},
            "average_steps_per_request": 0.0,
            "table_size_changes": [],
            "incomplete_requests": [],
            "completion_rate": 0.0,
            "validity_rate": 0.0,
            "generation_mode": generation_mode or "unknown",
        }

        try:
            total_actions = 0
            request_step_counts = []
            total_failures = 0
            validity_failures = 0

            for request_id, logs in self.log_entries.items():
                request_steps = 0
                has_end_action = False
                initial_size = None
                final_size = None

                for log in logs:
                    # Track generation modes
                    if log.generation_mode:
                        mode_count = summary_data["generation_mode_counts"].get(
                            log.generation_mode, 0
                        )
                        summary_data["generation_mode_counts"][log.generation_mode] = (
                            mode_count + 1
                        )

                    if log.action == "initial":
                        if log.table_summary:
                            initial_size = (
                                log.table_summary.get("num_rows", 0),
                                log.table_summary.get("num_columns", 0),
                            )
                    elif log.action.startswith("end"):
                        has_end_action = True
                        if log.table_summary:
                            final_size = (
                                log.table_summary.get("num_rows", 0),
                                log.table_summary.get("num_columns", 0),
                            )
                    elif not log.action.startswith("initial"):
                        if log.success:
                            total_actions += 1
                            request_steps += 1

                            # Count action types
                            action_type = (
                                log.action.split("(")[0]
                                if "(" in log.action
                                else log.action
                            )
                            summary_data["action_counts"][action_type] = (
                                summary_data["action_counts"].get(action_type, 0) + 1
                            )

                            if log.table_summary:
                                current_size = (
                                    log.table_summary.get("num_rows", 0),
                                    log.table_summary.get("num_columns", 0),
                                )
                                final_size = current_size
                        else:
                            total_failures += 1
                            if log.failure_type:
                                summary_data["failure_type_counts"][
                                    log.failure_type
                                ] = (
                                    summary_data["failure_type_counts"].get(
                                        log.failure_type, 0
                                    )
                                    + 1
                                )
                                if log.failure_type == "validity_failure":
                                    validity_failures += 1

                if request_steps > 0:
                    request_step_counts.append(request_steps)

                if not has_end_action:
                    summary_data["incomplete_requests"].append(request_id)

                # Track table size changes
                if initial_size and final_size:
                    size_change = {
                        "request_id": request_id,
                        "initial_size": initial_size,
                        "final_size": final_size,
                        "row_change": final_size[0] - initial_size[0],
                        "col_change": final_size[1] - initial_size[1],
                    }
                    summary_data["table_size_changes"].append(size_change)

            # Calculate derived metrics
            summary_data["total_transformations"] = total_actions
            summary_data["average_steps_per_request"] = (
                sum(request_step_counts) / len(request_step_counts)
                if request_step_counts
                else 0.0
            )
            summary_data["completion_rate"] = (
                1.0
                - (
                    len(summary_data["incomplete_requests"])
                    / summary_data["total_requests"]
                )
                if summary_data["total_requests"] > 0
                else 0.0
            )

            # Calculate validity rate
            total_attempts = total_actions + total_failures
            summary_data["validity_rate"] = (
                ((total_attempts - validity_failures) / total_attempts)
                if total_attempts > 0
                else 0.0
            )

        except Exception as e:
            summary_data["error"] = str(e)

        return summary_data

    def write_summary_report(self, generation_mode: str = None) -> None:
        """Write summary report to file"""
        summary_data = self.create_summary_report(generation_mode)

        try:
            summary_file = self.log_dir / "summary_report.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            print(f"Table transformation summary written to {summary_file}")
            self._print_summary_stats(summary_data)

        except Exception as e:
            print(f"Error creating summary report: {e}")

    def _print_summary_stats(self, summary_data: Dict[str, Any]) -> None:
        """Print summary statistics to console"""
        print(f"Generation mode: {summary_data.get('generation_mode', 'unknown')}")
        print(f"Total requests: {summary_data.get('total_requests', 0)}")
        print(f"Total transformations: {summary_data.get('total_transformations', 0)}")
        print(f"Completion rate: {summary_data.get('completion_rate', 0.0):.2%}")
        print(
            f"Average steps per request: {summary_data.get('average_steps_per_request', 0.0):.1f}"
        )
        print(
            f"Incomplete requests: {len(summary_data.get('incomplete_requests', []))}"
        )

        # Generation mode breakdown
        mode_counts = summary_data.get("generation_mode_counts", {})
        if mode_counts:
            print("Generation mode breakdown:")
            for mode, count in mode_counts.items():
                print(f"  {mode}: {count}")

        # Failure breakdown
        failure_counts = summary_data.get("failure_type_counts", {})
        if failure_counts:
            print("Failure breakdown:")
            total_failures = sum(failure_counts.values())
            for failure_type, count in failure_counts.items():
                percentage = (count / total_failures) * 100 if total_failures > 0 else 0
                print(f"  {failure_type}: {count} ({percentage:.1f}%)")

            validity_rate = summary_data.get("validity_rate", 0.0)
            print(f"Validity rate (parseable actions): {validity_rate:.1%}")

    def get_logging_stats(self) -> Dict[str, Any]:
        """Get current logging statistics"""
        return {
            "enabled": self.enable_logging,
            "log_dir": str(self.log_dir),
            "requests_logged": len(self.log_entries),
            "total_entries": sum(len(logs) for logs in self.log_entries.values()),
            "save_readable_tables": self.save_readable_tables,
            "log_format": self.log_format,
        }
