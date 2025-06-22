# Pyisolate Integration Test Suite

This directory contains comprehensive integration tests for the `pyisolate` library. The test suite focuses on end-to-end testing of the entire system across various scenarios and configurations.

## Test Structure

### `test_integration.py`
Main integration test file containing core functionality tests:

- **TestMultipleExtensionsWithConflictingDependencies**: Tests loading extensions with different/conflicting dependencies (e.g., numpy 1.x vs 2.x)
- **TestShareTorchConfiguration**: Tests `share_torch` configuration scenarios (both true and false)
- **TestHostExtensionInteraction**: Tests calling patterns between host and extensions
- **TestRecursiveCalling**: Tests recursive calling patterns (host→extension→host→extension)
- **TestComplexIntegrationScenarios**: Tests complex scenarios combining multiple features

### `test_edge_cases.py`
Edge cases and error handling tests:

- **TestExtensionErrors**: Tests error handling (missing dependencies, runtime errors)
- **TestConfigurationEdgeCases**: Tests edge cases in configuration (disabled extensions, malformed manifests)
- **TestConcurrentOperations**: Tests concurrent operations and thread safety
- **TestResourceManagement**: Tests resource cleanup and shutdown sequences

## Test Features

The test suite validates all the requirements specified:

1. ✅ **Loading multiple extensions in different venvs**
2. ✅ **Multiple extensions with conflicting dependencies** (numpy 1.x vs 2.x)
3. ✅ **Both share_torch: true and false scenarios**
4. ✅ **Calling extension functions from the host**
5. ✅ **Calling host functions from extensions** (via shared APIs)
6. ✅ **Recursive calling patterns**

## Running Tests

### Run All Integration Tests
```bash
# Using pytest directly
python -m pytest tests/test_integration.py -v

# Using the test runner script
python run_integration_tests.py
```

### Run Specific Test Categories
```bash
# Test conflicting dependencies
python -m pytest tests/test_integration.py::TestMultipleExtensionsWithConflictingDependencies -v

# Test share_torch configuration
python -m pytest tests/test_integration.py::TestShareTorchConfiguration -v

# Test host-extension interaction
python -m pytest tests/test_integration.py::TestHostExtensionInteraction -v

# Test recursive calling
python -m pytest tests/test_integration.py::TestRecursiveCalling -v

# Test edge cases
python -m pytest tests/test_edge_cases.py -v
```

### Run Individual Tests
```bash
# Test numpy version conflicts
python -m pytest tests/test_integration.py::TestMultipleExtensionsWithConflictingDependencies::test_numpy_version_conflicts -v

# Test share_torch=false
python -m pytest tests/test_integration.py::TestShareTorchConfiguration::test_share_torch_false -v
```

## Test Dependencies

The test suite only requires the dependencies already listed in `pyproject.toml`:
- `pytest>=7.0`
- `pytest-asyncio>=0.21.0`
- `pyyaml>=5.4.0`

**Important**: Extension-specific dependencies (torch, numpy, scipy, etc.) are **not** added to `pyproject.toml`. Instead, they are specified in the individual extension `manifest.yaml` files created during test execution. This ensures proper isolation and testing of the dependency management features.

## Test Architecture

### IntegrationTestBase
The `IntegrationTestBase` class provides common utilities for all integration tests:

- **Environment Setup**: Creates temporary test environments with proper directory structure
- **Extension Creation**: Dynamically creates test extensions with custom code and dependencies
- **Extension Loading**: Loads multiple extensions with different configurations
- **Cleanup**: Properly shuts down extensions and cleans up temporary resources

### Test Extensions
Tests create extensions dynamically with different characteristics:

- **Dependency variations**: numpy 1.x vs 2.x, torch with/without sharing, etc.
- **Isolation settings**: isolated vs shared environments
- **Custom functionality**: Extensions that test specific interaction patterns
- **Error scenarios**: Extensions that fail in various ways for error testing

### Shared Communication
Extensions communicate with the host and each other through:

- **DatabaseSingleton**: Shared database API for storing and retrieving test results
- **Extension APIs**: Direct method calls from host to extensions
- **Recursive patterns**: Complex calling chains to test deep interaction scenarios

## Notes

- **Torch tests may be slow**: Tests involving torch installation can take several minutes as torch is a large dependency
- **Temporary environments**: Each test creates its own isolated temporary environment that is cleaned up afterward
- **Real venv creation**: Tests create actual virtual environments to ensure realistic testing
- **Based on example/**: The test suite uses the same patterns and base classes as the provided example folder

## Coverage

The test suite provides comprehensive coverage of:
- ✅ Extension loading and initialization
- ✅ Dependency isolation and conflict resolution
- ✅ Configuration scenarios (share_torch, isolation, etc.)
- ✅ Host-extension communication patterns
- ✅ Error handling and edge cases
- ✅ Resource management and cleanup
- ✅ Concurrent operations and thread safety
