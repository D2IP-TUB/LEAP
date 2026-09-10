# Changelog

All notable changes to the LEAP project are documented in this file.

## [Unreleased] - 19.12.2025

### Added
 - Added xGrammar constrained decoding and retained legacy state-machine support.
 - Added automatic selection of isolated modern and legacy vLLM runtimes.
 - Introduced JSON operation output, strict parsing, and JSON Schema constraints.
 - Added Direct Query, NL2SQL, NL2Code, End2End, and CoT End2End answer extractors.
 - Added iterative, Chain-of-Table, and direct-query prompt configurations.
 - Expanded experiment matrices across strategies, constraints, backends, formats, models, and
   repeats.
 - Optimized experiment execution through model reuse, grouped sessions, health checks, and worker
   restarts.
 - Improved reports with per-extractor accuracy, aggregate metrics, failures, runtimes, and
   model-load details.
 - Fixed action parsing, prompt generation, global constraints, sorting/grouping, and add_column
   compatibility issues.
 - Updated GPU allocations and added an accuracy-analysis notebook.
#### Features
- **Action Registry System**: Implemented centralized action management system
  - New registry-based architecture for registering and managing actions
  - Dedicated action package structure with modular action classes
  - Registry-based action instantiation and management
- **Direct Query Action**: New action type for direct question answering
  - Automatically executed once after END action
  - Cannot be selected directly by the model, triggered programmatically
  - Best-effort parsing of query results and improved prompting
  - Dedicated prompt template for direct query action
- **Action History in Prompts**: Added action history context to prompts
- **Single Action Optimization**: Skip action type generation when only one action is available
- **Sampling Layer**: Implemented multi-sample generation with voting mechanism for improved action selection
  - Generates N candidate actions and selects winner via majority voting
  - Support for per-action sample configuration (e.g., different sample counts for different action types)
  - Extensible architecture with override points for custom generation, filtering, and aggregation logic
  - Debug mode for observing generation and voting process in real-time
  - Concurrent sample generation using `asyncio.gather()` for maximum throughput
- **Shuffle Invariant SamplingLayer**: Implement shuffle invariant sampling for an equivariance action behavior
- **Visual Logger Implementation**: Implemented a visual logger that helps visualise the output of the experiments
#### Refactoring
- Renamed `inference.py` to `types.py` for better clarity
- Moved `action.py` from `leap/core/` to `leap/core/actions/`
- Extract shared orchesteration for generation strategies
- Use sampling layer as an interface for single response requests
- Enhanced evaluator, prompt builder, and sampling logic
- Refactored inference constraints for better maintainability

### Changed
- Updated configuration schema in `configs/default.yaml` for action registry support

## [0.1.0] - 2025-12-03

### Added

#### Development Tools
- **Pre-commit Hooks**: Added pre-commit hooks to ensure code quality
- **Testing Infrastructure**: Introduced basic tests with pytest and pytest-cov
- **Profiler**: Added profiler to monitor generation bottlenecks (experimental)
- **UV Package Manager**: Migrated to `uv` for environment and dependency management

#### Features
- **Continuous Batching**: Adjusted and improved continuous batching for generation
- **Tensor Parallel Support**: Added tensor parallel support for distributed processing
- **Configuration**: Support for per-run and per-model configuration
- **Global Constraints**: Implemented global constraints as a configurable flag
- **Visual Logger**: Interactive visual logger for experiments 

### Changed
- **Prompt Builder**: Dedicated module for creating prompts

#### Refactoring (November-December 2025)
- **Generation Strategies**: Moved generation strategy (Iterative, CoTable-Style) logic to external modules

#### Code Quality
- **Formatting**: Applied consistent code formatting and linting with ruff (Dec 2, 2025)

#### Bug Fixes
- Graceful vLLM worker state cleanup on generation errors - workers keep running 

### Removed
- **Unused Configurations**