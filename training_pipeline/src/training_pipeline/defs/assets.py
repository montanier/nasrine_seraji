from pathlib import Path
from typing import Tuple

import pandas as pd
import dagster as dg
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from property_friends.models import CATEGORICAL_COLS, TARGET, training
from property_friends.data_input.factory import create_data_loader

train_dataset_path = "/data/datasets/train.csv"
test_dataset_path = "/data/datasets/test.csv"
serialized_dir = Path("/data/models/")


@dg.asset
def train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data_loader = create_data_loader(
        "filesystem", train_path=train_dataset_path, test_path=test_dataset_path
    )
    train_df = data_loader.load_training_data()
    train_columns = list(train_df.columns)
    train_columns.remove(TARGET)
    train_inputs = train_df[train_columns]
    train_cat_inputs = train_df[CATEGORICAL_COLS]
    train_target = train_df[TARGET]

    return train_inputs, train_cat_inputs, train_target


@dg.asset
def test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data_loader = create_data_loader(
        "filesystem", train_path=train_dataset_path, test_path=test_dataset_path
    )
    test_df = data_loader.load_test_data()

    train_columns = list(test_df.columns)
    train_columns.remove(TARGET)

    test_cat_inputs = test_df[CATEGORICAL_COLS]
    test_inputs = test_df[train_columns]
    test_target = test_df[TARGET]

    return test_inputs, test_cat_inputs, test_target


@dg.asset
def preprocessor(
    train_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
) -> TargetEncoder:
    train_inputs, train_cat_inputs, train_target = train_data
    return training.get_fitted_preprocessor(train_cat_inputs, train_target)


@dg.asset
def transformed_train(
    train_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    preprocessor: TargetEncoder,
) -> pd.DataFrame:
    train_inputs, train_cat_inputs, train_target = train_data
    train_cat_features = training.get_transformed_dataset(
        train_cat_inputs, preprocessor
    )
    train_inputs.loc[:, CATEGORICAL_COLS] = train_cat_features.values
    return train_inputs


@dg.asset
def transformed_test(
    test_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series], preprocessor: TargetEncoder
) -> pd.DataFrame:
    test_inputs, test_cat_inputs, test_target = test_data
    test_cat_features = training.get_transformed_dataset(test_cat_inputs, preprocessor)
    test_inputs.loc[:, CATEGORICAL_COLS] = test_cat_features.values
    return test_inputs


@dg.asset
def model(
    train_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    transformed_train: pd.DataFrame,
) -> GradientBoostingRegressor:

    _, _, train_target = train_data
    return training.get_trained_model(transformed_train, train_target)


@dg.asset
def evaluate(
    model: GradientBoostingRegressor,
    test_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    transformed_test: pd.DataFrame,
) -> Tuple[float, float, float]:

    _, _, test_target = test_data
    predictions = model.predict(transformed_test)
    return training.get_metrics(predictions, test_target.values)


@dg.asset
def serialize(model: GradientBoostingRegressor, preprocessor: TargetEncoder):
    serialized_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = Path(serialized_dir) / "preprocessor.joblib"
    training.serialize_preprocessor(preprocessor, preprocessor_path)
    model_path = Path(serialized_dir) / "model.joblib"
    training.serialize_preprocessor(model, model_path)
