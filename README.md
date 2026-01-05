## Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
source .venv/bin/activate
```

This will create a virtual environment for the project and install all required dependencies.

## Developer Setup

If you plan to contribute to the codebase, install the development dependencies and enable the pre-commit hooks:

```bash
uv sync --group dev
pre-commit install
```

This installs the linting/formatting tools defined in pyproject.toml and ensures they run automatically before each commit.
