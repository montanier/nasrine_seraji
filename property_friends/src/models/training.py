"""Machine learning model utilities for property price prediction.

This module provides functions for preprocessing categorical features and training
models for property valuation tasks.
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)


def get_fitted_preprocessor(
    categorical_features: pd.DataFrame, target: pd.DataFrame
) -> TargetEncoder:
    """Creates and fits a TargetEncoder for categorical feature preprocessing.

    Args:
        categorical_features: DataFrame containing categorical features to encode.
        target: DataFrame containing the target variable for target encoding.

    Returns:
        A fitted TargetEncoder instance ready for transforming categorical data.
    """
    encoder = TargetEncoder()
    encoder.fit(categorical_features, target)
    return encoder


def get_transformed_dataset(
    dataset: pd.DataFrame, encoder: TargetEncoder
) -> pd.DataFrame:
    """Transforms a dataset using a fitted TargetEncoder.

    Args:
        dataset: DataFrame containing categorical data to transform.
        encoder: A fitted TargetEncoder instance for transformation.

    Returns:
        DataFrame with categorical features transformed to numerical values.
    """
    return encoder.transform(dataset)


def get_trained_model(
    train_cols: pd.DataFrame, target: pd.DataFrame
) -> GradientBoostingRegressor:
    """Trains a GradientBoostingRegressor model with predefined hyperparameters.

    Args:
        train_cols: DataFrame containing the training features.
        target: DataFrame containing the training target values.

    Returns:
        A fitted GradientBoostingRegressor model ready for predictions.
    """
    model = GradientBoostingRegressor(
        **{
            "learning_rate": 0.1,
            "n_estimators": 500,
            "max_depth": 6,
            "subsample": 0.8,
            "loss": "huber",
        }
    )
    model.fit(train_cols, target.values.ravel())
    return model


def serialize_model(model: GradientBoostingRegressor, path: Union[str, Path]) -> None:
    """Serializes a trained model to a file.

    Args:
        model: The fitted GradientBoostingRegressor to serialize.
        path: File path where the model should be saved.
    """
    joblib.dump(model, path)


def load_model(path: Union[str, Path]) -> GradientBoostingRegressor:
    """Loads a serialized model from a file.

    Args:
        path: File path where the model is stored.

    Returns:
        The loaded GradientBoostingRegressor model.
    """
    return joblib.load(path)


def serialize_preprocessor(encoder: TargetEncoder, path: Union[str, Path]) -> None:
    """Serializes a fitted preprocessor to a file.

    Args:
        encoder: The fitted TargetEncoder to serialize.
        path: File path where the preprocessor should be saved.
    """
    joblib.dump(encoder, path)


def load_preprocessor(path: Union[str, Path]) -> TargetEncoder:
    """Loads a serialized preprocessor from a file.

    Args:
        path: File path where the preprocessor is stored.

    Returns:
        The loaded TargetEncoder preprocessor.
    """
    return joblib.load(path)


def get_metrics(
    predictions: Union[np.ndarray, list], target: Union[np.ndarray, list]
) -> Tuple[float, float, float]:
    """Calculates regression metrics for model evaluation.

    Args:
        predictions: Model predictions array or list.
        target: True target values array or list.

    Returns:
        Tuple containing (RMSE, MAPE, MAE) metrics.
    """
    rmse = np.sqrt(mean_squared_error(predictions, target))
    mape = mean_absolute_percentage_error(predictions, target)
    mae = mean_absolute_error(predictions, target)

    return rmse, mape, mae
