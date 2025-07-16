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
    # Create and train preprocessor
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "casa", "departamento"],
            "sector": ["providencia", "nunoa", "providencia", "nunoa"],
        }
    )
    target = pd.DataFrame({"price": [200000, 150000, 220000, 160000]})
    preprocessor = get_fitted_preprocessor(categorical_features, target)

    # Create and train model
    transformed_features = preprocessor.transform(categorical_features)
    model = get_trained_model(transformed_features, target)

    # Test dataset
    test_data = pd.DataFrame(
        {"type": ["casa", "departamento"], "sector": ["providencia", "nunoa"]}
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
    categorical_features = pd.DataFrame(
        {
            "type": ["casa", "departamento", "oficina"],
            "sector": ["providencia", "nunoa", "las_condes"],
        }
    )
    target = pd.DataFrame({"price": [300000, 200000, 500000]})

    # Train preprocessor and model
    preprocessor = get_fitted_preprocessor(categorical_features, target)
    transformed_features = preprocessor.transform(categorical_features)
    model = get_trained_model(transformed_features, target)

    # Test data
    test_data = pd.DataFrame({"type": ["casa"], "sector": ["providencia"]})

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
    categorical_features = pd.DataFrame(
        {"type": ["casa", "departamento"], "sector": ["providencia", "nunoa"]}
    )
    target = pd.DataFrame({"price": [250000, 180000]})

    # Train models
    preprocessor = get_fitted_preprocessor(categorical_features, target)
    transformed_features = preprocessor.transform(categorical_features)
    model = get_trained_model(transformed_features, target)

    test_data = pd.DataFrame({"type": ["casa"], "sector": ["providencia"]})

    # Direct prediction
    direct_prediction = model.predict(preprocessor.transform(test_data))

    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor_path = Path(temp_dir) / "preprocessor.joblib"
        model_path = Path(temp_dir) / "model.joblib"

        serialize_preprocessor(preprocessor, preprocessor_path)
        serialize_model(model, model_path)

        # When
        file_prediction = predict_from_files(preprocessor_path, model_path, test_data)

        # Then - should match direct prediction
        np.testing.assert_array_almost_equal(direct_prediction, file_prediction)
