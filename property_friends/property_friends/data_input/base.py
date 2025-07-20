"""Abstract base class for data loading operations."""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loading operations.

    This interface allows for different data sources (filesystem, database, cloud)
    while maintaining a consistent API for the ML pipeline.
    """

    @abstractmethod
    def load_training_data(self) -> pd.DataFrame:
        """Load training dataset.

        Returns:
            Training DataFrame.
        """
        pass

    @abstractmethod
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset.

        Returns:
            Test DataFrame.
        """
        pass

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both training and test datasets.

        Returns:
            Tuple containing the training and test DataFrames.
        """
        train_df = self.load_training_data()
        test_df = self.load_test_data()
        return train_df, test_df
