# pyisolate

**Run Python extensions in isolated virtual environments with seamless inter-process communication.**

> ⚠️ **Warning**: This library is currently in active development and the API may change. While the core functionality is working, it should not be considered stable for production use yet.

pyisolate enables you to run Python extensions with conflicting dependencies in the same application by automatically creating isolated virtual environments for each extension. Extensions communicate with the host process through a transparent RPC system, making the isolation invisible to your code.

## Key Benefits

- 🔒 **Dependency Isolation**: Run extensions with incompatible dependencies (e.g., numpy 1.x and 2.x) in the same application
- 🚀 **Zero-Copy PyTorch Tensor Sharing**: Share PyTorch tensors between processes without serialization overhead
- 🔄 **Transparent Communication**: Call async methods across process boundaries as if they were local
- 🎯 **Simple API**: Clean, intuitive interface with minimal boilerplate
- ⚡ **Fast**: Uses `uv` for blazing-fast virtual environment creation

## Installation

```bash
pip install pyisolate
```

For development:
```bash
pip install pyisolate[dev]
```

## Quick Start

### Basic Usage

Create an extension that runs in an isolated environment:

```python
# extensions/my_extension/__init__.py
from pyisolate import ExtensionBase

class MyExtension(ExtensionBase):
    def on_module_loaded(self, module):
        self.module = module

    async def process_data(self, data):
        # This runs in an isolated process with its own dependencies
        import numpy as np  # This could be numpy 2.x
        return np.array(data).mean()
```

Load and use the extension from your main application:

```python
# main.py
import pyisolate
import asyncio

async def main():
    # Configure the extension manager
    config = pyisolate.ExtensionManagerConfig(
        venv_root_path="./venvs"
    )
    manager = pyisolate.ExtensionManager(pyisolate.ExtensionBase, config)

    # Load an extension with specific dependencies
    extension = await manager.load_extension(
        pyisolate.ExtensionConfig(
            name="data_processor",
            module_path="./extensions/my_extension",
            isolated=True,
            dependencies=["numpy>=2.0.0"]
        )
    )

    # Use the extension
    result = await extension.process_data([1, 2, 3, 4, 5])
    print(f"Mean: {result}")  # Mean: 3.0

    # Cleanup
    await extension.stop()

asyncio.run(main())
```

### PyTorch Tensor Sharing

Share PyTorch tensors between processes without serialization:

```python
# extensions/ml_extension/__init__.py
from pyisolate import ExtensionBase
import torch

class MLExtension(ExtensionBase):
    async def process_tensor(self, tensor: torch.Tensor):
        # Tensor is shared, not copied!
        return tensor.mean()
```

```python
# main.py
extension = await manager.load_extension(
    pyisolate.ExtensionConfig(
        name="ml_processor",
        module_path="./extensions/ml_extension",
        share_torch=True  # Enable zero-copy tensor sharing
    )
)

# Large tensor is shared, not serialized
large_tensor = torch.randn(1000, 1000)
mean = await extension.process_tensor(large_tensor)
```

### Shared State with Singletons

Share state across all extensions using ProxiedSingleton:

```python
# shared.py
from pyisolate import ProxiedSingleton

class DatabaseAPI(ProxiedSingleton):
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
```

```python
# extensions/extension_a/__init__.py
class ExtensionA(ExtensionBase):
    async def save_result(self, result):
        db = DatabaseAPI()  # Returns proxy to host's instance
        await db.set("result", result)

# extensions/extension_b/__init__.py
class ExtensionB(ExtensionBase):
    async def get_result(self):
        db = DatabaseAPI()  # Returns proxy to host's instance
        return await db.get("result")
```

### Complete Application Structure

A complete pyisolate application requires a special `main.py` entry point to handle virtual environment activation:

```python
# main.py
if __name__ == "__main__":
    # When running as the main script, import and run your host application
    from host import main
    main()
else:
    # When imported by extension processes, ensure venv is properly activated
    import os
    import site
    import sys

    if os.name == "nt":  # Windows-specific venv activation
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv != "":
            sys.path.insert(0, os.path.join(venv, "Lib", "site-packages"))
            site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
```

```python
# host.py - Your main application logic
import pyisolate
import asyncio

async def async_main():
    # Create extension manager
    config = pyisolate.ExtensionManagerConfig(
        venv_root_path="./extension-venvs"
    )
    manager = pyisolate.ExtensionManager(ExtensionBase, config)

    # Load extensions (e.g., from a directory or configuration file)
    extensions = []
    for extension_path in discover_extensions():
        extension_config = pyisolate.ExtensionConfig(
            name=extension_name,
            module_path=extension_path,
            isolated=True,
            dependencies=load_dependencies(extension_path),
            apis=[SharedAPI]  # Optional shared singletons
        )
        extension = manager.load_extension(extension_config)
        extensions.append(extension)

    # Use extensions
    for extension in extensions:
        result = await extension.process()
        print(f"Result: {result}")

    # Clean shutdown
    for extension in extensions:
        await extension.stop()

def main():
    asyncio.run(async_main())
```

This structure ensures that:
- The host application runs normally when executed directly
- Extension processes properly activate their virtual environments when spawned
- Windows-specific path handling is properly managed

## Features

### Core Features
- **Automatic Virtual Environment Management**: Creates and manages isolated environments automatically
- **Bidirectional RPC**: Extensions can call host methods and vice versa
- **Async/Await Support**: Full support for asynchronous programming
- **Lifecycle Hooks**: `before_module_loaded()`, `on_module_loaded()`, and `stop()` for setup/teardown
- **Error Propagation**: Exceptions are properly propagated across process boundaries

### Advanced Features
- **Dependency Resolution**: Automatically installs extension-specific dependencies
- **Platform Support**: Works on Windows, Linux, and soon to be tested on macOS
- **Context Tracking**: Ensures callbacks happen on the same asyncio loop as the original call
- **Fast Installation**: Uses `uv` for 10-100x faster package installation without every extension having its own copy of libraries

## Architecture

```
┌─────────────────────┐     RPC      ┌─────────────┐
│    Host Process     │◄────────────►│ Extension A │
│                     │              │  (venv A)   │
│  ┌──────────────┐   │              └─────────────┘
│  │   Shared     │   │     RPC      ┌─────────────┐
│  │ Singletons   │   │◄────────────►│ Extension B │
│  └──────────────┘   │              │  (venv B)   │
└─────────────────────┘              └─────────────┘
```

## Roadmap

### ✅ Completed
- [x] Core isolation and RPC system
- [x] Automatic virtual environment creation
- [x] Bidirectional communication
- [x] PyTorch tensor sharing
- [x] Shared singleton pattern
- [x] Comprehensive test suite
- [x] Windows, Linux support
- [x] Security features (path normalization)
- [x] Fast installation with `uv`
- [x] Context tracking for RPC calls
- [x] Async/await support

### 🚧 In Progress
- [ ] Documentation site
- [ ] macOS testing
- [ ] Performance benchmarks
- [ ] Wrapper for non-async calls between processes

### 🔮 Future Plans
- [ ] Network access restrictions per extension
- [ ] Filesystem access sandboxing
- [ ] CPU/Memory usage limits
- [ ] Hot-reloading of extensions
- [ ] Distributed RPC (across machines)
- [ ] Profiling and debugging tools

## Use Cases

pyisolate is perfect for:

- **Plugin Systems**: When plugins may require conflicting dependencies
- **ML Pipelines**: Different models requiring different library versions
- **Microservices in a Box**: Multiple services with different dependencies in one app
- **Testing**: Running tests with different dependency versions in parallel
- **Legacy Code Integration**: Wrapping legacy code with specific dependency requirements

## Development

We welcome contributions!

```bash
# Setup development environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"
pre-commit install

# Run tests
pytest

# Run linting
ruff check pyisolate tests
```

## License

pyisolate is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on Python's `multiprocessing` and `asyncio`
- Uses [`uv`](https://github.com/astral-sh/uv) for fast package management
- Inspired by plugin systems like Chrome Extensions and VS Code Extensions

---

**Star this repo** if you find it useful! ⭐
