"""Machine learning model utilities for property price prediction.

This module provides functions for preprocessing categorical features and training
models for property valuation tasks.
"""

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor


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
