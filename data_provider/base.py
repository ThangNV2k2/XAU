from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Common interface so the underlying market data source can be swapped later."""

    @abstractmethod
    def get_historical(
        self,
        interval: str,
        outputsize: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by timestamp (ascending) with columns: open, high, low, close."""

    @abstractmethod
    def get_latest_price(self) -> float:
        """Return the latest traded price as a float."""
