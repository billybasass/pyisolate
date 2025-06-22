import logging
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import override

from pyisolate import ExtensionBase, ProxiedSingleton

logger = logging.getLogger(__name__)


class DatabaseSingleton(ProxiedSingleton):
    def __init__(self):
        self._db: dict[str, Any] = {}

    async def set_value(self, key: str, value: Any) -> None:
        self._db[key] = value

    async def get_value(self, key: str) -> Any:
        return self._db.get(key, None)


class ExampleExtension(ABC):
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def prepare_shutdown(self):
        """Prepare for shutdown, e.g., clean up resources."""

    @abstractmethod
    async def do_stuff(self, value: Any) -> Any:
        """An example method that does something."""


class ExampleExtensionBase(ExtensionBase):
    """An example extension base class."""

    @override
    async def before_module_loaded(self) -> None:
        """Do any setup that must be done before loading the extension."""

    @override
    async def on_module_loaded(self, module) -> None:
        """Once the module itself is loaded, access the entrypoint."""
        if not getattr(module, "example_entrypoint", None):
            raise RuntimeError(f"Module {module.__name__} does not have an 'example_entrypoint' function.")

        # Call the entrypoint function from the module
        extension: ExampleExtension = module.example_entrypoint()
        if not isinstance(extension, ExampleExtension):
            raise TypeError(f"Module {module.__name__} did not return an instance of ExampleExtension.")
        self.extension = extension

        await extension.initialize()

        logger.debug(f"ExampleExtension: Module {module.__name__} loaded successfully.")

    async def do_stuff(self, value) -> Any:
        """An example method that does something."""
        return await self.extension.do_stuff(value)
