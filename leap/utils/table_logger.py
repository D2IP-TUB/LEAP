"""
Table Logger Module - Extracted table logging functionality

This module handles all table transformation logging, analysis, and reporting
functionality that was previously embedded in the main processing logic.
"""

import csv
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from leap.core import Table


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
    
    def __init__(self, 
                 log_dir: str = "table_logs",
                 enable_logging: bool = True,
                 save_readable_tables: bool = True,
                 compress_logs: bool = False,
                 log_format: str = "readable",
                 max_table_chars: int = 10000):
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
    
    def serialize_table_to_csv(self, table: Table, max_chars: int = None) -> str:
        """Convert table to CSV string with limited characters."""
        if max_chars is None:
            max_chars = self.max_table_chars

        return table.to_csv(max_chars=max_chars, max_rows=10, crop=True)
    
    def get_table_summary(self, table: Table) -> Dict[str, Any]:
        """Get a compact summary of table dimensions."""
        num_rows, num_cols = table.get_size()
        columns = list(table.columns)

        return {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "columns": columns[:5] + ["..."] if len(columns) > 5 else columns
        }
    
    def log_table_state(self,
                       request_id: str,
                       step: int,
                       action: str,
                       table: Table,
                       success: bool = True,
                       failure_type: Optional[str] = None,
                       generation_mode: Optional[str] = None) -> None:
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
        """
        if not self.enable_logging:
            return
        
        try:
            # Create log entry
            table_summary = self.get_table_summary(table)
            table_preview = None
            
            # Create readable table representation
            if self.save_readable_tables:
                table_csv = self.serialize_table_to_csv(table, max_chars=self.max_table_chars)
                table_preview = table_csv.split('\n')[:6]  # First 5 rows + header
                
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
                table_summary=table_summary,
                table_preview=table_preview
            )
            
            # Store in memory for analysis
            if request_id not in self.log_entries:
                self.log_entries[request_id] = []
            self.log_entries[request_id].append(log_entry)
            
            # Write to file
            self._write_log_entry(log_entry)
            
        except Exception as e:
            print(f"Warning: Failed to log table state: {e}")
    
    def _save_table_csv(self, request_id: str, step: int, action: str, table: Dict[str, Any]) -> None:
        """Save full table as CSV file"""
        try:
            # Clean action name for filename
            clean_action = self._clean_filename(action)
            table_filename = f"{request_id}_step{step:02d}_{clean_action}.csv"
            table_path = self.log_dir / table_filename
            
            with open(table_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(table['columns'])
                for row in table['rows']:
                    writer.writerow(row)
        except Exception as e:
            print(f"Warning: Failed to save table CSV: {e}")
    
    def _clean_filename(self, action: str) -> str:
        """Clean action string for use in filename"""
        clean_action = action.replace('(', '_').replace(')', '').replace('[', '').replace(']', '')
        clean_action = clean_action.replace('"', '').replace(',', '_').replace(' ', '_')
        return clean_action[:50]  # Limit length
    
    def _write_log_entry(self, log_entry: LogEntry) -> None:
        """Write log entry to file"""
        try:
            log_filename = f"{log_entry.request_id}_log.json"
            log_path = self.log_dir / log_filename
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry.to_dict(), indent=2) + '\n' + '-'*80 + '\n')
        except Exception as e:
            print(f"Warning: Failed to write log entry: {e}")
    
    def set_request_metadata(self, request_id: str, metadata: Dict[str, Any]) -> None:
        """Set metadata for a request"""
        self.request_metadata[request_id] = metadata
    
    def get_request_logs(self, request_id: str) -> List[LogEntry]:
        """Get all log entries for a specific request"""
        return self.log_entries.get(request_id, [])
    
    def analyze_request(self, request_id: str) -> Dict[str, Any]:
        """Analyze logs for a specific request"""
        logs = self.get_request_logs(request_id)
        if not logs:
            return {"error": "No logs found for request"}
        
        analysis = {
            "request_id": request_id,
            "total_steps": len(logs),
            "successful_steps": sum(1 for log in logs if log.success),
            "failed_steps": sum(1 for log in logs if not log.success),
            "generation_modes": list(set(log.generation_mode for log in logs if log.generation_mode)),
            "failure_types": [log.failure_type for log in logs if log.failure_type],
            "actions_taken": [log.action for log in logs],
            "duration": logs[-1].timestamp - logs[0].timestamp if len(logs) > 1 else 0.0,
            "completed": any(log.action.startswith('end') for log in logs)
        }
        
        # Table size progression
        table_sizes = []
        for log in logs:
            if log.table_summary:
                size = (log.table_summary.get('num_rows', 0), log.table_summary.get('num_columns', 0))
                table_sizes.append(size)
        analysis["table_size_progression"] = table_sizes
        
        return analysis
    
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
            "generation_mode": generation_mode or "unknown"
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
                        mode_count = summary_data["generation_mode_counts"].get(log.generation_mode, 0)
                        summary_data["generation_mode_counts"][log.generation_mode] = mode_count + 1
                    
                    if log.action == 'initial':
                        if log.table_summary:
                            initial_size = (log.table_summary.get('num_rows', 0), 
                                          log.table_summary.get('num_columns', 0))
                    elif log.action.startswith('end'):
                        has_end_action = True
                        if log.table_summary:
                            final_size = (log.table_summary.get('num_rows', 0), 
                                        log.table_summary.get('num_columns', 0))
                    elif not log.action.startswith('initial'):
                        if log.success:
                            total_actions += 1
                            request_steps += 1
                            
                            # Count action types
                            action_type = log.action.split('(')[0] if '(' in log.action else log.action
                            summary_data["action_counts"][action_type] = summary_data["action_counts"].get(action_type, 0) + 1
                            
                            if log.table_summary:
                                current_size = (log.table_summary.get('num_rows', 0), 
                                              log.table_summary.get('num_columns', 0))
                                final_size = current_size
                        else:
                            total_failures += 1
                            if log.failure_type:
                                summary_data["failure_type_counts"][log.failure_type] = summary_data["failure_type_counts"].get(log.failure_type, 0) + 1
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
                        "col_change": final_size[1] - initial_size[1]
                    }
                    summary_data["table_size_changes"].append(size_change)
            
            # Calculate derived metrics
            summary_data["total_transformations"] = total_actions
            summary_data["average_steps_per_request"] = (
                sum(request_step_counts) / len(request_step_counts) 
                if request_step_counts else 0.0
            )
            summary_data["completion_rate"] = (
                1.0 - (len(summary_data["incomplete_requests"]) / summary_data["total_requests"])
                if summary_data["total_requests"] > 0 else 0.0
            )
            
            # Calculate validity rate
            total_attempts = total_actions + total_failures
            summary_data["validity_rate"] = (
                ((total_attempts - validity_failures) / total_attempts) 
                if total_attempts > 0 else 0.0
            )
            
        except Exception as e:
            summary_data["error"] = str(e)
        
        return summary_data
    
    def write_summary_report(self, generation_mode: str = None) -> None:
        """Write summary report to file"""
        summary_data = self.create_summary_report(generation_mode)
        
        try:
            summary_file = self.log_dir / "summary_report.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
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
        print(f"Average steps per request: {summary_data.get('average_steps_per_request', 0.0):.1f}")
        print(f"Incomplete requests: {len(summary_data.get('incomplete_requests', []))}")
        
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
    
    def analyze_table_logs(self, request_id: str) -> None:
        """Analyze and print table logs for a specific request"""
        if not self.enable_logging:
            print("Table logging is disabled")
            return
        
        logs = self.get_request_logs(request_id)
        if not logs:
            # Try to load from file
            logs = self._load_logs_from_file(request_id)
        
        if not logs:
            print(f"No logs found for request: {request_id}")
            return
        
        print(f"\nTable transformation log for {request_id}:")
        print("=" * 80)
        
        for log in logs:
            status_str = "SUCCESS" if log.success else f"FAILED ({log.failure_type})"
            mode_str = f" [{log.generation_mode}]" if log.generation_mode else ""
            print(f"\nStep {log.step}: {log.action} - {status_str}{mode_str}")
            
            if log.table_summary:
                rows = log.table_summary.get('num_rows', 0)
                cols = log.table_summary.get('num_columns', 0)
                print(f"Table: {rows} rows × {cols} columns")
            
            if log.table_preview:
                print("Preview:")
                for line in log.table_preview:
                    print(f"  {line}")
            
            # Mention CSV file location
            if log.step > 0 and self.save_readable_tables:
                csv_files = list(self.log_dir.glob(f"{request_id}_step{log.step:02d}_*.csv"))
                if csv_files:
                    print(f"Full table saved as: {csv_files[0].name}")
            
            print("-" * 40)
    
    def _load_logs_from_file(self, request_id: str) -> List[LogEntry]:
        """Load logs from file for a specific request"""
        try:
            log_file = self.log_dir / f"{request_id}_log.json"
            if not log_file.exists():
                return []
            
            logs = []
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                blocks = content.split('-'*80)
                
                for block in blocks:
                    block = block.strip()
                    if block:
                        try:
                            entry_data = json.loads(block)
                            log_entry = LogEntry(**entry_data)
                            logs.append(log_entry)
                        except (json.JSONDecodeError, TypeError):
                            continue
            
            return sorted(logs, key=lambda x: x.step)
        except Exception as e:
            print(f"Error loading logs from file: {e}")
            return []
    
    def list_table_files_for_request(self, request_id: str) -> List[str]:
        """List all table CSV files for a specific request"""
        if not self.enable_logging or not self.save_readable_tables:
            return []
        
        try:
            files = list(self.log_dir.glob(f"{request_id}_*.csv"))
            return sorted([f.name for f in files])
        except Exception:
            return []
    
    def cleanup_logs(self, older_than_hours: int = 24) -> int:
        """Clean up old log files"""
        if not self.enable_logging:
            return 0
        
        cutoff_time = time.time() - (older_than_hours * 3600)
        removed_count = 0
        
        try:
            for file_path in self.log_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return removed_count
    
    def get_logging_stats(self) -> Dict[str, Any]:
        """Get current logging statistics"""
        return {
            "enabled": self.enable_logging,
            "log_dir": str(self.log_dir),
            "requests_logged": len(self.log_entries),
            "total_entries": sum(len(logs) for logs in self.log_entries.values()),
            "save_readable_tables": self.save_readable_tables,
            "log_format": self.log_format
        }


# Factory functions for common configurations
def create_production_logger(log_dir: str = "table_logs") -> TableLogger:
    """Create logger for production use with minimal overhead"""
    return TableLogger(
        log_dir=log_dir,
        enable_logging=True,
        save_readable_tables=False,  # Reduced I/O
        compress_logs=True,
        log_format="json",
        max_table_chars=1000
    )


def create_debug_logger(log_dir: str = "table_logs") -> TableLogger:
    """Create logger for debugging with full details"""
    return TableLogger(
        log_dir=log_dir,
        enable_logging=True,
        save_readable_tables=True,
        compress_logs=False,
        log_format="readable",
        max_table_chars=10000
    )


def create_analysis_logger(log_dir: str = "table_logs") -> TableLogger:
    """Create logger optimized for post-hoc analysis"""
    return TableLogger(
        log_dir=log_dir,
        enable_logging=True,
        save_readable_tables=True,
        compress_logs=False,
        log_format="json",
        max_table_chars=5000
    )


# Context manager for automatic logging
class LoggingContext:
    """Context manager for request-scoped logging"""
    
    def __init__(self, logger: TableLogger, request_id: str, metadata: Dict[str, Any] = None):
        self.logger = logger
        self.request_id = request_id
        self.metadata = metadata or {}
        self.step_counter = 0
    
    def __enter__(self):
        self.logger.set_request_metadata(self.request_id, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.log_table_state(
                self.request_id, 
                self.step_counter,
                f"context_error: {exc_type.__name__}",
                {"columns": [], "rows": []},  # Empty table for error case
                success=False,
                failure_type="context_error"
            )
    
    def log_step(self, action: str, table: Dict[str, Any], success: bool = True, 
                failure_type: str = None, generation_mode: str = None):
        """Log a step within this context"""
        self.logger.log_table_state(
            self.request_id, 
            self.step_counter,
            action, 
            table,
            success=success,
            failure_type=failure_type,
            generation_mode=generation_mode
        )
        self.step_counter += 1


if __name__ == "__main__":
    # Example usage
    logger = create_debug_logger()
    
    # Example table
    sample_table = {
        "columns": ["Name", "Age", "City"],
        "rows": [
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
            ["Charlie", "35", "Chicago"]
        ]
    }
    
    # Example logging
    with LoggingContext(logger, "test_request", {"question": "Test question"}) as ctx:
        ctx.log_step("initial", sample_table, generation_mode="test")
        ctx.log_step("select_column([\"Name\", \"Age\"])", 
                    {"columns": ["Name", "Age"], "rows": [["Alice", "25"], ["Bob", "30"]]},
                    generation_mode="test")
        ctx.log_step("end()", 
                    {"columns": ["Name", "Age"], "rows": [["Alice", "25"], ["Bob", "30"]]},
                    generation_mode="test")
    
    # Generate report
    logger.write_summary_report("test")
    
    # Analyze specific request
    logger.analyze_table_logs("test_request")