from typing import Tuple

import pandas as pd


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test data into pandas DataFrames
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
