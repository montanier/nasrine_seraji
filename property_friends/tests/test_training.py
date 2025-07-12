import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from property_friends.models.training import (
    get_fitted_preprocessor,
    get_transformed_dataset,
    get_trained_model,
    serialize_model,
    load_model,
    serialize_preprocessor,
    load_preprocessor,
    get_metrics,
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


def test_get_trained_model_basic() -> None:
    # Given
    train_cols = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    target = pd.DataFrame({"price": [100000, 150000, 200000, 250000, 300000]})

    # When
    model = get_trained_model(train_cols, target)

    # Then
    assert isinstance(model, GradientBoostingRegressor)


def test_get_trained_model_can_predict() -> None:
    # Given
    train_cols = pd.DataFrame(
        {"size": [100, 150, 200, 120, 180], "rooms": [2, 3, 4, 2, 3]}
    )
    target = pd.DataFrame({"price": [200000, 300000, 400000, 220000, 350000]})
    test_data = pd.DataFrame({"size": [130, 170], "rooms": [2, 3]})

    # When
    model = get_trained_model(train_cols, target)
    predictions = model.predict(test_data)

    # Then
    assert len(predictions) == 2
    assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
    assert all(pred > 0 for pred in predictions)  # Prices should be positive


def test_serialize_and_load_model() -> None:
    # Given
    train_cols = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
    )
    target = pd.DataFrame({"price": [100, 200, 300, 400, 500]})
    model = get_trained_model(train_cols, target)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.joblib"

        # When
        serialize_model(model, model_path)
        loaded_model = load_model(model_path)

        # Then
        assert isinstance(loaded_model, GradientBoostingRegressor)
        assert loaded_model.learning_rate == model.learning_rate
        assert loaded_model.n_estimators == model.n_estimators

        # Verify predictions are the same
        original_pred = model.predict(train_cols)
        loaded_pred = loaded_model.predict(train_cols)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


def test_serialize_and_load_preprocessor() -> None:
    # Given
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa"],
            "sector": ["providencia", "nunoa", "providencia"],
        }
    )
    target = pd.DataFrame({"price": [100000, 80000, 110000]})
    encoder = get_fitted_preprocessor(categorical_features, target)

    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_path = Path(temp_dir) / "test_encoder.joblib"

        # When
        serialize_preprocessor(encoder, encoder_path)
        loaded_encoder = load_preprocessor(encoder_path)

        # Then
        assert isinstance(loaded_encoder, TargetEncoder)

        # Verify transformations are the same
        original_transform = encoder.transform(categorical_features)
        loaded_transform = loaded_encoder.transform(categorical_features)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)


def test_get_metrics_basic() -> None:
    # Given
    predictions = [10.0, 20.0, 30.0, 40.0]
    target = [12.0, 18.0, 32.0, 38.0]

    # When
    rmse, mape, mae = get_metrics(predictions, target)

    # Then
    # Expected calculations:
    # Errors: [-2, 2, -2, 2]
    # MSE: (4 + 4 + 4 + 4) / 4 = 4
    # RMSE: sqrt(4) = 2.0
    assert rmse == 2.0

    # MAE: (2 + 2 + 2 + 2) / 4 = 2.0
    assert mae == 2.0

    # MAPE: (2/12 + 2/18 + 2/32 + 2/38) / 4
    expected_mape = (2 / 12 + 2 / 18 + 2 / 32 + 2 / 38) / 4
    assert abs(mape - expected_mape) < 1e-10

    # Check types
    assert isinstance(rmse, float)
    assert isinstance(mape, float)
    assert isinstance(mae, float)
