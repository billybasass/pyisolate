import logging

import numpy as np
import pandas as pd
from shared import DatabaseSingleton, ExampleExtension
from typing_extensions import override

logger = logging.getLogger(__name__)
db = DatabaseSingleton()


class Extension1(ExampleExtension):
    """Extension using pandas and numpy 1.x for data processing."""

    @override
    async def initialize(self):
        logger.debug("Extension1 initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("Extension1 preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        logger.debug("Extension1 processing data with pandas and numpy")

        # Create a DataFrame with some data
        data = {"name": ["Alice", "Bob", "Charlie"], "score": [95, 87, 92]}
        df = pd.DataFrame(data)

        # Use numpy for calculations
        mean_score = np.mean(df["score"])
        numpy_version = np.__version__
        pandas_version = pd.__version__

        result = {
            "extension": "extension1",
            "data_rows": len(df),
            "mean_score": float(mean_score),
            "numpy_version": numpy_version,
            "pandas_version": pandas_version,
            "input_value": value,
        }

        # Store result in shared database
        await db.set_value("extension1_result", result)

        return f"Extension1 processed {len(df)} rows with mean score {mean_score}"


def example_entrypoint() -> ExampleExtension:
    """Entrypoint function for the extension."""
    return Extension1()
