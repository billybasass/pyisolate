import logging

import numpy as np
import scipy.stats as stats
from bs4 import BeautifulSoup
from shared import DatabaseSingleton, ExampleExtension
from typing_extensions import override

logger = logging.getLogger(__name__)
db = DatabaseSingleton()


class Extension3(ExampleExtension):
    """Extension using beautifulsoup4 and scipy for web scraping and statistics."""

    @override
    async def initialize(self):
        logger.debug("Extension3 initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("Extension3 preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        logger.debug("Extension3 parsing HTML and computing statistics")

        # Mock HTML content
        html_content = """
        <html>
            <body>
                <div class="data">42</div>
                <div class="data">58</div>
                <div class="data">73</div>
                <div class="data">91</div>
                <div class="data">67</div>
            </body>
        </html>
        """

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        data_elements = soup.find_all("div", class_="data")
        values = [int(elem.text) for elem in data_elements]

        # Use scipy and numpy for statistical analysis
        mean_val = np.mean(values)
        median_val = float(np.median(values))
        mode_result = stats.mode(values, keepdims=True)

        result = {
            "extension": "extension3",
            "parsed_values": values,
            "count": len(values),
            "mean": float(mean_val),
            "median": median_val,
            "mode": float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None,
            "input_value": value,
            "soup_title": soup.title.string if soup.title else None,
        }

        # Store result in shared database
        await db.set_value("extension3_result", result)

        return f"Extension3 parsed {len(values)} values with mean {mean_val}"


def example_entrypoint() -> ExampleExtension:
    """Entrypoint function for the extension."""
    return Extension3()
