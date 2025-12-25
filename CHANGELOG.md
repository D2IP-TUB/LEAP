# Changelog

All notable changes to the LEAP project are documented in this file.

## [Unreleased] - 19.12.2025

### Added

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
- **Action Examples System**: New comprehensive action examples framework
  - YAML-based action examples configuration (`configs/action_examples.yaml`)
- **Configurable Generation Strategies**: Added support for reading generation strategy from config
  - Support for direct strategy selection via configuration
  - Add query only generation strategy

#### Configuration
- **Enhanced Prompt Configuration**: Added configurable prompt length settings
- **Temperature Configuration**: Added per-task temperature configuration for action type selection
- **Model Configuration**: Extended model configuration in `configs/models.yaml`
- **Action Examples Configuration**: New `configs/action_examples.yaml` for managing action examples

#### Refactoring
- Renamed `inference.py` to `types.py` for better clarity
- Moved `action.py` from `leap/core/` to `leap/core/actions/`
- Modularized action implementations:
  - `leap/core/actions/end.py` - END action implementation
  - `leap/core/actions/select_column.py` - SELECT_COLUMN action
  - `leap/core/actions/select_row.py` - SELECT_ROW action
  - `leap/core/actions/direct_query.py` - Direct query action
- Extract shared orchestration for generation strategies
- Use sampling layer as an interface for single response requests
- Enhanced evaluator, prompt builder, and sampling logic
- Refactored inference constraints for better maintainability
- **Prompt Builder Enhancements**:
  - Improved tag placement for examples and conditions (use tokenizer for automatic prompt building)
  - Better handling of action examples in prompts
  - Support for action history context

### Changed
- Updated configuration schema in `configs/default.yaml` for action registry support
- Enhanced prompt templates with better example formatting
- Improved action type selection prompt with additional examples
- Reduced temperature for action type selection for more consistent results (similar to CoTables paper)
- Replaced `use_chain_of_table` config flag with generation strategy `cot`
- Generation strategies are now uniformly configured through the strategy selection in config
- Direct query action now reads examples from `action_examples.yaml`

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