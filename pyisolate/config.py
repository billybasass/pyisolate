from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ._internal.shared import ProxiedSingleton


class ExtensionManagerConfig(TypedDict):
    """Configuration for the ExtensionManager.

    This configuration controls the behavior of the ExtensionManager, which is
    responsible for creating and managing multiple extensions.

    Example:
        >>> config = ExtensionManagerConfig(
        ...     venv_root_path="/path/to/extension-venvs"
        ... )
        >>> manager = ExtensionManager(MyExtensionBase, config)
    """

    venv_root_path: str
    """The root directory where virtual environments for isolated extensions will be created.

    Each extension gets its own subdirectory under this path. The path should be writable
    and have sufficient space for installing dependencies.
    """


class ExtensionConfig(TypedDict):
    """Configuration for a specific extension.

    This configuration defines how an individual extension should be loaded and
    managed by the ExtensionManager. It controls isolation, dependencies, and
    shared resources.

    Example:
        >>> config = ExtensionConfig(
        ...     name="data_processor",
        ...     module_path="./extensions/processor",
        ...     isolated=True,
        ...     dependencies=["numpy>=1.26.0", "pandas>=2.0.0"],
        ...     apis=[DatabaseAPI, ConfigAPI],
        ...     share_torch=False
        ... )
        >>> extension = manager.load_extension(config)
    """

    name: str
    """A unique name for this extension.

    This will be used as the directory name for the virtual environment (after
    normalization for filesystem safety). Should be descriptive and unique within
    your application.
    """

    module_path: str
    """The filesystem path to the extension package directory.

    This must be a directory containing an __init__.py file. The path can be
    absolute or relative to the current working directory.
    """

    isolated: bool
    """Whether to run this extension in an isolated virtual environment.

    If True, a separate venv is created with the specified dependencies.
    If False, the extension runs in the host Python environment.
    """

    dependencies: list[str]
    """List of pip-installable dependencies for this extension.

    Each string should be a valid pip requirement specifier (e.g.,
    "numpy>=1.21.0", "requests~=2.28.0"). Dependencies are installed
    in the order specified.

    Security Note: The ExtensionManager validates dependencies to prevent
    command injection, but you should still review dependency lists from
    untrusted sources.
    """

    apis: list[type[ProxiedSingleton]]
    """List of ProxiedSingleton classes that this extension should have access to.

    These singletons will be automatically configured to use remote instances
    from the host process, enabling shared state across all extensions.
    """

    share_torch: bool
    """Whether to share PyTorch with the host process.

    If True, the extension will use torch.multiprocessing for process creation
    and the exact same PyTorch version as the host. This enables zero-copy
    tensor sharing between processes. If False, the extension can install its
    own PyTorch version if needed.
    """
