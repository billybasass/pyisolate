"""Memory benchmark extension base class."""

from typing import Any

from pyisolate import ExtensionBase


class MemoryBenchmarkExtensionBase(ExtensionBase):
    """Base class for memory benchmark extensions."""

    async def before_module_loaded(self) -> None:
        """Do any setup that must be done before loading the extension."""

    async def on_module_loaded(self, module) -> None:
        """Once the module itself is loaded, access the entrypoint."""
        if not getattr(module, "memory_benchmark_entrypoint", None):
            raise RuntimeError(
                f"Module {module.__name__} does not have a 'memory_benchmark_entrypoint' function."
            )

        # Call the entrypoint function from the module
        extension = module.memory_benchmark_entrypoint()
        self.extension = extension

        await extension.initialize()

    async def do_stuff(self, value: Any) -> Any:
        """Process a value."""
        return await self.extension.do_stuff(value)

    async def store_tensor(self, tensor_id: str, tensor: Any) -> dict[str, Any]:
        """Store a tensor and return memory info."""
        return await self.extension.store_tensor(tensor_id, tensor)

    async def clear_tensors(self) -> None:
        """Clear all stored tensors."""
        return await self.extension.clear_tensors()

    async def get_tensor_info(self, tensor_id: str) -> dict[str, Any]:
        """Get info about a stored tensor."""
        return await self.extension.get_tensor_info(tensor_id)
