import pandas as pd
from category_encoders import TargetEncoder


def get_fitted_preprocessor(
    categorical_features: pd.DataFrame, target: pd.DataFrame
) -> TargetEncoder:
    encoder = TargetEncoder()
    encoder.fit(categorical_features, target)
    return encoder


def get_transformed_dataset(
    dataset: pd.DataFrame, encoder: TargetEncoder
) -> pd.DataFrame:
    return encoder.transform(dataset)
