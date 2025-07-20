import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from property_friends.models.training import (
    get_fitted_preprocessor,
    get_trained_model,
    serialize_model,
    serialize_preprocessor,
)
from property_friends.models.prediction import predict_from_files


def test_predict_from_files_basic() -> None:
    # Given
    # Create training data with all features
    training_data = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa", "departamento"],
            "sector": ["providencia", "nunoa", "providencia", "nunoa"],
            "size": [100, 80, 120, 90],  # Add numerical features
            "price": [200000, 150000, 220000, 160000],
        }
    )

    categorical_features = training_data[["type", "sector"]]
    target = training_data[["price"]]
    preprocessor = get_fitted_preprocessor(categorical_features, target)

    # Create and train model with full feature set
    full_features = training_data.drop("price", axis=1)
    full_features.loc[:, ["type", "sector"]] = preprocessor.transform(
        categorical_features
    ).values
    model = get_trained_model(full_features, target)

    # Test dataset (should match training feature structure minus target)
    test_data = pd.DataFrame(
        {
            "type": ["casa", "departamento"],
            "sector": ["providencia", "nunoa"],
            "size": [110, 85],
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor_path = Path(temp_dir) / "preprocessor.joblib"
        model_path = Path(temp_dir) / "model.joblib"

        # Serialize both
        serialize_preprocessor(preprocessor, preprocessor_path)
        serialize_model(model, model_path)

        # When
        predictions = predict_from_files(preprocessor_path, model_path, test_data)

        # Then
        assert len(predictions) == 2
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
        assert all(pred > 0 for pred in predictions)  # Prices should be positive


def test_predict_from_files_consistency() -> None:
    # Given
    training_data = pd.DataFrame(
        {
            "type": ["casa", "departamento", "oficina"],
            "sector": ["providencia", "nunoa", "las_condes"],
            "size": [150, 100, 200],
            "price": [300000, 200000, 500000],
        }
    )

    categorical_features = training_data[["type", "sector"]]
    target = training_data[["price"]]
    preprocessor = get_fitted_preprocessor(categorical_features, target)

    # Train model with full feature set
    full_features = training_data.drop("price", axis=1)
    full_features.loc[:, ["type", "sector"]] = preprocessor.transform(
        categorical_features
    ).values
    model = get_trained_model(full_features, target)

    # Test data
    test_data = pd.DataFrame(
        {"type": ["casa"], "sector": ["providencia"], "size": [140]}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor_path = Path(temp_dir) / "preprocessor.joblib"
        model_path = Path(temp_dir) / "model.joblib"

        serialize_preprocessor(preprocessor, preprocessor_path)
        serialize_model(model, model_path)

        # When - make predictions twice
        predictions1 = predict_from_files(preprocessor_path, model_path, test_data)
        predictions2 = predict_from_files(preprocessor_path, model_path, test_data)

        # Then - should be identical
        np.testing.assert_array_equal(predictions1, predictions2)


def test_predict_from_files_matches_direct_prediction() -> None:
    # Given
    training_data = pd.DataFrame(
        {
            "type": ["casa", "departamento"],
            "sector": ["providencia", "nunoa"],
            "size": [120, 90],
            "price": [250000, 180000],
        }
    )

    categorical_features = training_data[["type", "sector"]]
    target = training_data[["price"]]
    preprocessor = get_fitted_preprocessor(categorical_features, target)

    # Train model with full feature set
    full_features = training_data.drop("price", axis=1)
    full_features.loc[:, ["type", "sector"]] = preprocessor.transform(
        categorical_features
    ).values
    model = get_trained_model(full_features, target)

    test_data = pd.DataFrame(
        {"type": ["casa"], "sector": ["providencia"], "size": [115]}
    )

    # Direct prediction (transform categorical features only)
    test_data_direct = test_data.copy()
    test_data_direct.loc[:, ["type", "sector"]] = preprocessor.transform(
        test_data[["type", "sector"]]
    ).values
    direct_prediction = model.predict(test_data_direct)

    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor_path = Path(temp_dir) / "preprocessor.joblib"
        model_path = Path(temp_dir) / "model.joblib"

        serialize_preprocessor(preprocessor, preprocessor_path)
        serialize_model(model, model_path)

        # When
        file_prediction = predict_from_files(preprocessor_path, model_path, test_data)

        # Then - should match direct prediction
        np.testing.assert_array_almost_equal(direct_prediction, file_prediction)
