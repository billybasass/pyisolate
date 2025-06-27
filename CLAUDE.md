# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyisolate** is a Python library for running extensions across multiple isolated virtual environments with RPC communication. It solves dependency conflicts by isolating extensions in separate venvs while maintaining seamless host-extension communication through AsyncRPC.

## Development Commands

### Environment Setup
```bash
# Preferred (using uv - much faster)
uv venv && source .venv/bin/activate && uv pip install -e ".[dev,docs]"
pre-commit install

# With benchmarking dependencies
uv pip install -e ".[dev,docs,bench]"

# Alternative (using pip)
python -m venv venv && source venv/bin/activate && pip install -e ".[dev,docs]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyisolate --cov-report=html --cov-report=term-missing

# Run integration tests
pytest tests/test_integration.py -v

# Test the working example
cd example && python main.py -v
```

### Benchmarking
```bash
# Install benchmark dependencies
uv pip install -e ".[bench]"

# Run full benchmark suite
python benchmark.py

# Quick benchmarks (fewer iterations)
python benchmark.py --quick

# Skip torch benchmarks
python benchmark.py --no-torch

# Run benchmarks via pytest
pytest tests/test_benchmarks.py -v -s
```

### Code Quality
```bash
# Lint and format
ruff check pyisolate tests
ruff format pyisolate tests

# All quality checks
tox -e lint
```

### Build
```bash
# Build package
python -m build

# Build docs
cd docs && make html
```

## Architecture

### Core Components
- **ExtensionManager**: Manages multiple extensions and their isolated venvs
- **ExtensionBase**: Base class for extensions with lifecycle hooks (`before_module_loaded`, `on_module_loaded`)
- **AsyncRPC**: Inter-process communication system with context-aware call tracking
- **ProxiedSingleton**: Enables shared APIs across processes (like `DatabaseSingleton` in example)

Note: The example uses a two-level pattern where `ExampleExtensionBase` handles the lifecycle hooks and creates actual extension instances that implement `initialize()`, `prepare_shutdown()`, and custom methods like `do_stuff()`.

### Key Directories
- `pyisolate/`: Main package with public API in `__init__.py`
- `pyisolate/_internal/`: Core implementation (RPC, process management)
- `example/`: Working demo with 3 extensions showcasing dependency conflicts
- `tests/`: Integration and edge case tests

### Extension Workflow
1. ExtensionManager creates isolated venv per extension
2. Installs extension-specific dependencies
3. Launches extension in separate process via `_internal/client.py`
4. Establishes bidirectional RPC communication
5. Extensions can call host methods and shared singletons transparently

## Configuration

### Python Support
3.9 - 3.12 (tested in CI)

### Dependencies
- **Runtime**: None (pure Python)
- **Development**: pytest, ruff, pre-commit
- **Testing**: torch>=2.0.0, numpy (for `share_torch` and tensor tests)
- **Benchmarking**: torch, numpy, psutil, tabulate (for performance measurement)

### Key Config Files
- `pyproject.toml`: Project metadata, dependencies, tool configuration
- `tox.ini`: Multi-Python testing environments
- `.pre-commit-config.yaml`: Git hooks for code quality

## Special Features

### Dependency Isolation
Each extension gets its own venv - handles conflicting packages like numpy 1.x vs 2.x (demonstrated in example).

### PyTorch Sharing
Use `share_torch: true` in extension config to share PyTorch models across processes for memory efficiency.

### RPC Patterns
- Extensions can call host methods recursively (host→extension→host)
- Shared singletons work transparently via RPC proxying
- Context tracking prevents circular calls

## Testing Notes

The test suite covers real venv creation, dependency conflicts, and RPC edge cases. The `example/` directory provides a working demonstration with 3 extensions that showcase the core functionality.
