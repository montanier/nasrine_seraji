import pytest
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from property_friends.models.model import (
    get_fitted_preprocessor,
    get_transformed_dataset,
)


def test_get_fitted_preprocessor_basic() -> None:
    # Given
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa", "departamento"],
            "sector": ["providencia", "nunoa", "providencia", "nunoa"],
        }
    )
    target = pd.DataFrame({"price": [100000, 80000, 110000, 85000]})

    # When
    encoder = get_fitted_preprocessor(categorical_features, target)

    # Then
    assert isinstance(encoder, TargetEncoder)
    assert hasattr(encoder, "mapping")
    assert encoder.mapping is not None


def test_get_fitted_preprocessor_transform_consistency() -> None:
    # Given
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa", "departamento"],
            "sector": ["providencia", "nunoa", "providencia", "nunoa"],
        }
    )
    target = pd.DataFrame({"price": [100000, 80000, 110000, 85000]})

    # When
    encoder = get_fitted_preprocessor(categorical_features, target)
    transformed1 = encoder.transform(categorical_features)
    transformed2 = encoder.transform(categorical_features)

    # Then
    pd.testing.assert_frame_equal(transformed1, transformed2)


def test_get_fitted_preprocessor_with_nan_values() -> None:
    # Given
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", np.nan, "departamento"],
            "sector": ["providencia", "nunoa", "providencia", np.nan],
        }
    )
    target = pd.DataFrame({"price": [100000, 80000, 110000, 85000]})

    # When
    encoder = get_fitted_preprocessor(categorical_features, target)
    transformed = encoder.transform(categorical_features)

    # Then
    assert isinstance(encoder, TargetEncoder)
    assert transformed.shape == categorical_features.shape


def test_get_transformed_dataset_basic() -> None:
    # Given
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa", "departamento"],
            "sector": ["providencia", "nunoa", "providencia", "nunoa"],
        }
    )
    target = pd.DataFrame({"price": [100000, 80000, 110000, 85000]})
    encoder = get_fitted_preprocessor(categorical_features, target)
    dataset = pd.DataFrame(
        {"type": ["casa", "departamento"], "sector": ["providencia", "nunoa"]}
    )

    # When
    transformed = get_transformed_dataset(dataset, encoder)

    # Then
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == dataset.shape
    assert list(transformed.columns) == list(dataset.columns)


def test_get_transformed_dataset_with_unseen_values() -> None:
    # Given
    categorical_features = pd.DataFrame({"type": ["casa", "departamento"]})
    target = pd.DataFrame({"price": [100000, 80000]})
    encoder = get_fitted_preprocessor(categorical_features, target)
    dataset = pd.DataFrame(
        {"type": ["casa", "departamento", "oficina"]}  # 'oficina' is unseen
    )

    # When
    transformed = get_transformed_dataset(dataset, encoder)

    # Then
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == dataset.shape
    assert "type" in transformed.columns
    assert (
        transformed["type"][2] == 90000.0
    )  # unseen values should be at average of target


def test_get_transformed_dataset_with_unknown_columns() -> None:
    # Given
    categorical_features = pd.DataFrame({"type": ["casa", "departamento"]})
    target = pd.DataFrame({"price": [100000, 80000]})
    encoder = get_fitted_preprocessor(categorical_features, target)
    dataset = pd.DataFrame({"unknown_col": ["value1", "value2"]})

    # When / Then
    with pytest.raises(KeyError):
        get_transformed_dataset(dataset, encoder)
