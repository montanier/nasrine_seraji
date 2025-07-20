"""File system utilities for loading data from CSV files."""

from typing import Tuple
import pandas as pd
from .base import DataLoader


class FileSystemDataLoader(DataLoader):
    """Filesystem-based data loader for CSV files."""

    def __init__(self, train_path: str, test_path: str):
        """Initialize the filesystem data loader.

        Args:
            train_path: Path to the training CSV file.
            test_path: Path to the test CSV file.
        """
        self.train_path = train_path
        self.test_path = test_path

    def load_training_data(self) -> pd.DataFrame:
        """Load training data from CSV file.

        Returns:
            Training DataFrame.
        """
        return pd.read_csv(self.train_path)

    def load_test_data(self) -> pd.DataFrame:
        """Load test data from CSV file.

        Returns:
            Test DataFrame.
        """
        return pd.read_csv(self.test_path)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy function for backward compatibility.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.

    Returns:
        Tuple containing the training and test DataFrames.
    """
    loader = FileSystemDataLoader(train_path, test_path)
    return loader.load_data()
