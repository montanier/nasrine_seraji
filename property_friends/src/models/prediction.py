"""Prediction utilities for property price estimation.

This module provides functions for making predictions using serialized models
and preprocessors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from .model import load_model, load_preprocessor


def predict_from_files(
    preprocessor_path: Union[str, Path],
    model_path: Union[str, Path],
    dataset: pd.DataFrame,
) -> np.ndarray:
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

    # Transform features and make predictions
    transformed_data = preprocessor.transform(dataset)
    predictions = model.predict(transformed_data)

    return np.asarray(predictions)
