"""Prediction utilities for property price estimation.

This module provides functions for making predictions using serialized models
and preprocessors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from numpy.typing import NDArray
from .training import load_model, load_preprocessor
from property_friends.models import CATEGORICAL_COLS, TARGET


def predict_from_files(
    preprocessor_path: Union[str, Path],
    model_path: Union[str, Path],
    dataset: pd.DataFrame,
) -> NDArray[np.float64]:
    """Makes predictions using serialized preprocessor and model.

    Args:
        preprocessor_path: Path to the serialized TargetEncoder preprocessor.
        model_path: Path to the serialized GradientBoostingRegressor model.
        dataset: DataFrame containing the features to predict on.

    Returns:
        Array of predictions from the model.
    """
    # Load preprocessor and model
    preprocessor = load_preprocessor(preprocessor_path)
    model = load_model(model_path)

    # Transform categorical features
    input_columns = list(dataset.columns)
    if TARGET in input_columns:
        input_columns.remove(TARGET)
    transformed_data = preprocessor.transform(dataset[CATEGORICAL_COLS])
    dataset.loc[:, CATEGORICAL_COLS] = transformed_data.values

    # Make predictions
    predictions = model.predict(dataset)

    return np.asarray(predictions, dtype=np.float64)
