"""File system utilities for loading data from CSV files."""

from typing import Tuple

import pandas as pd


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the train and test data into pandas DataFrames.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.

    Returns:
        Tuple containing the training and test DataFrames.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
