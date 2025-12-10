# Changelog

All notable changes to the LEAP project are documented in this file.

## [Unreleased] - 10.12.2025

### Added

#### Features
- **Sampling Layer**: Implemented multi-sample generation with voting mechanism for improved action selection
  - Generates N candidate actions and selects winner via majority voting
  - Support for per-action sample configuration (e.g., different sample counts for different action types)
  - Extensible architecture with override points for custom generation, filtering, and aggregation logic
  - Debug mode for observing generation and voting process in real-time
  - Concurrent sample generation using `asyncio.gather()` for maximum throughput
- **Shuffle Invariant SamplingLayer**: Implement shuffle invariant sampling for an equivariance action behavior

#### Refactoring
  - Extract shared orchesteration for generation strategies
  - Use sampling layer as an interface for single response requests

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