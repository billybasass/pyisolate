import logging

import numpy as np
from shared import DatabaseSingleton, ExampleExtension
from typing_extensions import override

logger = logging.getLogger(__name__)
db = DatabaseSingleton()


class Extension2(ExampleExtension):
    """Extension using numpy 2.x and requests for HTTP operations."""

    @override
    async def initialize(self):
        logger.debug("Extension2 initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("Extension2 preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        logger.debug("Extension2 simulating HTTP request with numpy 2.x")

        # Simulate API response data
        mock_response = {"status": "success", "data": [1.5, 2.3, 3.1, 4.7, 5.9]}

        # Use numpy 2.x for array operations
        arr = np.array(mock_response["data"])
        stats = {
            "extension": "extension2",
            "numpy_version": np.__version__,
            "array_sum": float(np.sum(arr)),
            "array_mean": float(np.mean(arr)),
            "array_std": float(np.std(arr)),
            "input_value": value,
            "simulated_request": True,
        }

        # Store result in shared database
        await db.set_value("extension2_result", stats)

        return f"Extension2 processed array with sum {stats['array_sum']}"


def example_entrypoint() -> ExampleExtension:
    """Entrypoint function for the extension."""
    return Extension2()
